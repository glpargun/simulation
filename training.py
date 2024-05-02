import pandas as pd
import numpy as np
import optax
import torch
import os
import matplotlib.pyplot as plt
from flax import linen as nn
import flax.serialization
import flax.serialization
import flax.serialization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from jax import random, grad, jit
import jax.numpy as jnp


class PowerFlowTraining:
    def __init__(self, config_loader, config):
        self.config_loader = config_loader
        self.config = config
        self.key = random.PRNGKey(self.config_loader['random_state'])
        self.loader = Loader(config_loader)
        self.data, self.scalers, self.train_loader = self.loader.init_data()
        self.NN_model = self.init_PF_model()
        self.optimizer, self.params, self.opt_state = self.init_PF_optimizer()

    def init_PF_model(self):
        input_size = self.data['X_train_jax'].shape[1]
        output_size = self.data['y_train_jax'].shape[1]
        hidden_size = self.config['hidden_size']
        NN_model = SimpleNN(input_size, hidden_size, output_size)
        return NN_model
    
    def init_PF_optimizer(self):
        optimizer = optax.adam(learning_rate=self.config['lr'])
        params = self.NN_model.init(self.key, 
                                    jnp.ones([1, 6]),
                                    jnp.ones([1, self.NN_model.input_size-6]))['params']
        opt_state = optimizer.init(params)
        return optimizer, params, opt_state
    
    def init_functions(self):    
        @jit
        def power_flow(params, P_Q_gen, input_fixed):
            preds_scaled = self.NN_model.apply({'params': params}, P_Q_gen, input_fixed)
            return preds_scaled
        
        @jit
        def loss_PF(params, P_G, input_fixed, targets):
            preds = self.NN_model.apply({'params': params}, P_G, input_fixed)
            return jnp.mean((preds - targets) ** 2)
        
        @jit
        def update_PF(params, opt_state, P_Q_gen, input_fixed, targets):
            grads = grad(loss_PF)(params, P_Q_gen, input_fixed, targets)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state
        
        self.power_flow, self.loss_PF, self.update_PF = power_flow, loss_PF, update_PF

    def train(self, num_epochs, patience=10):
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0  # Counter to track how long the val_loss has stopped improving

        for epoch in range(num_epochs):
            losses = []
            for X, y in self.train_loader:
                X_jax, y_jax = jnp.array(X.numpy()), jnp.array(y.numpy())
                self.params, self.opt_state = self.update_PF(self.params, self.opt_state, X_jax[:, :6], X_jax[:, 6:], y_jax)
                losses.append(self.loss_PF(self.params, X_jax[:, :6], X_jax[:, 6:], y_jax))
            
            avg_train_loss = np.mean(losses)
            avg_val_loss = self.loss_PF(self.params, self.data['X_test_jax'][:, :6], self.data['X_test_jax'][:, 6:], self.data['y_test_jax'])
            
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Test Loss: {avg_val_loss}")

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0  # Reset the counter if there's improvement
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
                break

        return history
    
    def save_params(self, path='models/params_PF.bin'):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Serialize the parameters
            bytes_data = flax.serialization.to_bytes(self.params)
            
            # Debugging: Check if the bytes data is empty
            if not bytes_data:
                print("Warning: Serialized data is empty.")
            else:
                print(f"Serialized data size: {len(bytes_data)} bytes")
            
            # Write the serialized data to file
            with open(path, 'wb') as f:
                f.write(bytes_data)
            print("Parameters saved successfully.")
        except Exception as e:
            print(f"Failed to save parameters: {e}")

    def load_params(self, path='models/params_PF.bin'):
        try:
            with open(path, 'rb') as f:
                bytes_data = f.read()
            # Properly re-initialize the model before loading parameters
            dummy_input = jnp.ones([1, self.NN_model.input_size])  # Adjust according to the actual input size of your model
            self.params = self.NN_model.init(self.key, dummy_input)['params']  # Re-initialize parameters
            # Load the parameters from bytes
            self.params = flax.serialization.from_bytes(self.params, bytes_data)
            print("Parameters loaded successfully.")
        except Exception as e:
            print(f"Failed to load parameters: {e}")

def get_PF_functions(model, optimizer_PF):
    @jit
    def power_flow(params, P_Q_G, input_fixed):
        preds_scaled = model.apply({'params': params}, P_Q_G, input_fixed)
        return preds_scaled
    
    @jit
    def loss_PF(params, P_G, input_fixed, targets):
        preds = model.apply({'params': params}, P_G, input_fixed)
        return jnp.mean((preds - targets) ** 2)
    
    @jit
    def update_PF(params, opt_state, P_Q_G, input_fixed, targets):
        grads = grad(loss_PF)(params, P_Q_G, input_fixed, targets)
        updates, opt_state = optimizer_PF.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return loss_PF, update_PF, power_flow

class Loader:
    def __init__(self, config):
        self.features_path = config['path_features']
        self.target_path = config['path_target']
        self.test_size = config['test_size']
        self.random_state = config['random_state']
        self.batch_size = config['batch_size']

    def init_data(self):
        features = pd.read_csv(self.features_path)
        target = pd.read_csv(self.target_path)
        self.column_features = features.columns
        self.column_target = target.columns

        X_train, X_test, y_train, y_test = train_test_split(features, 
                                                            target, 
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        
        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler().fit(y_train)
        y_train_scaled = scaler_y.transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        data = {
            'X_train_jax': jnp.array(X_train_scaled),
            'y_train_jax': jnp.array(y_train_scaled),
            'X_test_jax': jnp.array(X_test_scaled),
            'y_test_jax': jnp.array(y_test_scaled)
        }

        scalers = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }

        train_dataset = TensorDataset(torch.tensor(X_train_scaled.astype(np.float32)),
                                      torch.tensor(y_train_scaled.astype(np.float32)))
        
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True)
        
        return data, scalers, train_loader

class SimpleNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int

    def setup(self):
        self.fc1 = nn.Dense(self.hidden_size)
        self.fc2 = nn.Dense(self.output_size)

    def __call__(self, x1, x2):
        x = jnp.concatenate([x1, x2], axis=-1)
        x = self.fc1(x)
        x = nn.tanh(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    config_loader_PF = {
        'path_features': 'data/X_PF.csv', 
        'path_target': 'data/Y_PF.csv', 
        'test_size': 0.2, 
        'batch_size': 64, 
        'random_state': 42
    }
    config_NN_PF = {
        'power_flow_model': 'model_simple', 
        'hidden_size': 50, 
        'lr': 0.00001, 
        'num_epochs': 5000
    }
    model = PowerFlowTraining(config_loader_PF, config_NN_PF)
    model.init_functions()

    print("Model parameter keys:", model.params.keys())
    print("Nested keys example:", {k: v.keys() for k, v in model.params.items()})
    print("Checking parameter integrity before saving...")
    if 'Dense_0' in model.params:
        print("Parameter example (first few weights):", model.params['Dense_0']['kernel'][:5])
    else:
        print("Expected key 'Dense_0' not found in parameters.")
    model.save_params()
    history = model.train(config_NN_PF['num_epochs'])
    y_pred_scaled = model.power_flow(model.params, model.data['X_test_jax'][:, :6], model.data['X_test_jax'][:, 6:])
    y_pred = model.scalers['scaler_y'].inverse_transform(y_pred_scaled)
    y_true = model.scalers['scaler_y'].inverse_transform(model.data['y_test_jax'])
    plt.plot(y_pred[:500, 0])
    plt.plot(y_true[:500, 0])
    plt.show()
