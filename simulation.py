import numpy as np
import pickle
import pandapower as pp
import pandas as pd
import pandapower.plotting as plot

from conf.bus_config import location_buses
from conf.connection_config import line_connections
from conf.line_config import line_data

class Simulation:
    def __init__(self, num_buses=160, max_battery_capacity=40):
        self.net = pp.create_empty_network()
        self.bus_index_name_mapping = {}
        self.num_buses = num_buses
        self.battery_soc = {bus_index: 0.5 for bus_index in range(num_buses)}
        self.battery_capacity = {bus_index: max_battery_capacity for bus_index in range(num_buses)}
        self.load_demand = None
        self.generation = None

    def load_data(self, demand_path, generation_path):
        with open(demand_path, 'rb') as f:
            self.load_demand = pickle.load(f)
        with open(generation_path, 'rb') as f:
            self.generation = pickle.load(f)     

    def variables(self, lower_P, upper_P, lower_Q, upper_Q):
        P_L = np.random.uniform(lower_P, upper_P) 
        Q_L = np.random.uniform(lower_Q, upper_Q) 
        P_G = np.random.uniform(lower_P, upper_P) 
        S_G = P_G 
        return P_L, Q_L, P_G, S_G
    
    def update_qg(self, S_G, lower_bound=0.9):
        P_G = np.random.uniform(lower_bound*S_G, S_G)
        sigma = np.random.choice([-1, 1])
        Q_G = sigma * np.sqrt(max(0, S_G**2 - P_G**2))
        return Q_G, P_G

    def create_std_line(self):
        pp.create_std_type(self.net, line_data, name="Al/St 240/40 4-bundle 380.0", element='line')

    def initialize_network(self):
        for key, value in location_buses.items():
            bus_index = pp.create_bus(self.net, vn_kv=380., name=f"Bus {key}", geodata=(value[0], value[1]), zone=value[2], max_vm_pu=np.inf, min_vm_pu=0.0)
            self.bus_index_name_mapping[bus_index] = f"Bus {key}"
            if value[2] == 'extgrid':
                pp.create_ext_grid(self.net, bus=bus_index, name=key)
            else:
                pp.create_sgen(self.net, bus=bus_index, p_mw=0, q_mvar=0, name=f"sgen {key}")
    
    def initialize_connection(self):
        for line in line_connections:
            from_bus = line["from_bus"]
            to_bus = line["to_bus"]
            length_km = line.get("length_km", 1)
            std_type = line["std_type"]
            num_parallel = max(line.get("num_parallel", 1), 1)
            from_bus_idx = pp.get_element_index(self.net, "bus", f"Bus {from_bus}")
            to_bus_idx = pp.get_element_index(self.net, "bus", f"Bus {to_bus}")
            pp.create_line(self.net, from_bus=from_bus_idx, to_bus=to_bus_idx, length_km=length_km, std_type=std_type, parallel=num_parallel)

    def update_sgens_with_battery(self):
        battery_rate = np.random.choice([-0.1, 0.1])
        soc_min = 0.3
        soc_max = 0.7
        for bus_index in range(self.num_buses):
            sgens_at_bus = self.net.sgen[self.net.sgen.bus == bus_index]
            if not sgens_at_bus.empty:
                sgen_id = sgens_at_bus.index[0]
                current_soc = self.battery_soc[bus_index]
                battery_capacity_mwh = self.battery_capacity[bus_index]
                if current_soc < soc_min:
                    charge_amount = battery_rate * battery_capacity_mwh
                    self.net.sgen.at[sgen_id, 'p_mw'] += charge_amount
                    current_soc += charge_amount / battery_capacity_mwh
                elif current_soc > soc_max:
                    discharge_amount = battery_rate * battery_capacity_mwh
                    self.net.sgen.at[sgen_id, 'p_mw'] -= discharge_amount
                    current_soc -= discharge_amount / battery_capacity_mwh
                self.battery_soc[bus_index] = min(max(current_soc, soc_min), soc_max)

    def add_loads(self):
        for bus_index in self.bus_index_name_mapping.keys():
            P_L, _, Q_L, _ = self.variables(10, 20, 5, 10)
            pp.create_load(self.net, bus=bus_index, p_mw=P_L, q_mvar=Q_L, name=f"Load at {self.bus_index_name_mapping[bus_index]}")

    def update_hourly_changes(self):
        for idx, bus_idx in enumerate(self.net.load.index):
            Q_L, _, P_L, _ = self.variables(5, 10, 10, 20)
            self.net.load.at[bus_idx, 'p_mw'] = P_L
            self.net.load.at[bus_idx, 'q_mvar'] = Q_L
        for idx, bus_idx in enumerate(self.net.sgen.index):
            _, _, P_G, S_G = self.variables(10, 30, 0, 0) 
            Q_G, P_G = self.update_qg(S_G)
            self.net.sgen.at[bus_idx, 'p_mw'] = P_G
            self.net.sgen.at[bus_idx, 'q_mvar'] = Q_G

    def run_simulation(self):
        self.update_sgens_with_battery()
        pp.runpp(self.net)
        P_line = self.net.res_line['p_from_mw'].values
        Q_line = self.net.res_line['q_from_mvar'].values
        V = self.net.res_bus['vm_pu'].values
        target = np.hstack([P_line, Q_line, V])[np.newaxis, :]
        P_G = self.net.sgen['p_mw'].values
        Q_G = self.net.sgen['q_mvar'].values
        P_L = self.net.load['p_mw'].values
        Q_L = self.net.load['q_mvar'].values
        inputs = np.hstack([P_G, Q_G, P_L, Q_L])[np.newaxis, :]
        return target, inputs

    def create_dataset(self, days=2):
        X, Y = [], []
        print(f"Starting dataset creation for {days} days, each with 24 hours.")
        for day in range(days):
            for hour in range(24):
                print(f"Day {day + 1}, Hour {hour + 1}: Updating loads and generations.")
                self.update_hourly_changes()  # Update network conditions for the current hour
                target, inputs = self.run_simulation()
                Y.append(target)
                X.append(inputs)
            print(f"Completed simulations for Day {day + 1}.")
        self.Y = np.vstack(Y)
        self.X = np.vstack(X)
        print("Dataset creation complete.")

    def save_dataset(self):
        print("Saving dataset to CSV")
        X_cols, Y_cols = self.get_columns()
        Y_df = pd.DataFrame(data=self.Y, columns=Y_cols)
        X_df = pd.DataFrame(data=self.X, columns=X_cols)
        try:
            Y_df.to_csv('./data/Y_PF.csv', index=False)
            X_df.to_csv('./data/X_PF.csv', index=False)
            print("Datasets saved successfully")
        except Exception as e:
            print(f"Failed to save datasets: {e}")

    def get_columns(self):
        X_cols, Y_cols = [], []
        for i in range(len(self.net.sgen)):
            X_cols.append(f"P_G{i}")
            X_cols.append(f"Q_G{i}")
        for i in range(len(self.net.bus)):
            X_cols.append(f"P_L{i}")
            X_cols.append(f"Q_L{i}")
        for idx, (i, j) in enumerate(self.net.line[['from_bus', 'to_bus']].values):
            Y_cols.append(f"P_{i}_{j}")
            Y_cols.append(f"Q_{i}_{j}")
        for i in range(len(self.net.bus)):
            Y_cols.append(f"V_{i}")
        return X_cols, Y_cols

if __name__ == '__main__':
    config_data = {
        'demand_data': './data/demand.pickle',
        'generation_data': './data/generation.pickle'
    }
    sim = Simulation()
    sim.create_std_line()
    sim.initialize_network()
    sim.initialize_connection()
    sim.add_loads()
    sim.create_dataset()
    sim.save_dataset()
