import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.plotting as plot

from conf.bus_config import location_buses
from conf.connection_config import line_connections
from conf.line_config import line_data

class Simulation:
    def __init__(self, num_buses=160, max_battery_capacity=40, eta_ch=0.95, eta_dch=0.95, E_max=1.0):
        self.net = pp.create_empty_network()
        self.bus_index_name_mapping = {}
        self.num_buses = num_buses
        self.battery_soc = {bus_index: 0.5 for bus_index in range(num_buses)}
        self.battery_capacity = {bus_index: max_battery_capacity for bus_index in range(num_buses)}
        self.eta_ch = eta_ch
        self.eta_dch = eta_dch
        self.E_max = E_max
        self.load_demand = None
        self.generation = None
        self.updated_P_G = []

    def load_data(self, demand_path, generation_path):
        try:
            self.load_demand = pd.read_csv(demand_path, skiprows=2)
            self.generation = pd.read_csv(generation_path, skiprows=2)

            # Convert kWh to MWh
            self.load_demand /= 1000
            self.generation /= 1000
        except Exception as e:
            print(f"Failed to load data from CSV: {e}")
            return

        try:
            num_groups = len(self.load_demand.columns) // 4
            if len(self.load_demand.columns) % 4 != 0:
                num_groups += 1 
            
            self.load_demand = self.load_demand.T.groupby(np.arange(num_groups)).mean().T
            self.generation = self.generation.T.groupby(np.arange(num_groups)).mean().T
        except Exception as e:
            print(f"Error in data aggregation: {e}")

    def update_qg(self, S_G, lower_bound=0.9):
        P_G = np.random.uniform(lower_bound*S_G, S_G)
        sigma = np.random.choice([-1, 1])
        Q_G = sigma * np.sqrt(max(0, S_G**2 - P_G**2))
        return Q_G, P_G

    def create_std_line(self):
        pp.create_std_type(self.net, line_data, name="Al/St 240/40 4-bundle 380.0", element='line')

    def initialize_network(self):
        print("Starting network initialization...")
        created_buses = 0
        for key, value in location_buses.items():
            try:
                bus_index = pp.create_bus(self.net, 
                                        vn_kv=380., 
                                        name=f"Bus {key}", 
                                        geodata=(value[0], value[1]), 
                                        zone=value[2], 
                                        max_vm_pu=np.inf, 
                                        min_vm_pu=0.0)
                self.bus_index_name_mapping[bus_index] = f"Bus {key}"
                created_buses += 1
                print(f"Bus {key} created with index {bus_index}.")
                if value[2] == 'extgrid':
                    pp.create_ext_grid(self.net, bus=bus_index, name=key)
                    print(f"External grid connected to Bus {key}.")
                else:
                    pp.create_sgen(self.net, bus=bus_index, p_mw=0, q_mvar=0, name=f"sgen {key}")
                    print(f"Static generator connected to Bus {key}.")
            except Exception as e:
                print(f"Failed to create bus {key}: {e}")

        print("Completed network initialization.")
        print(f"Total buses intended: {len(location_buses)}")
        print(f"Total buses successfully created: {created_buses}")
        print(f"Total buses in network: {len(self.net.bus)}")

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

    def update_battery_soc_and_pg(self, current_hour):
        P_ch_max = 0.1  # Max charging power
        P_dch_max = 0.1  # Max discharging power
        soc_min = 0.1
        soc_max = 0.7
        
        updated_P_G = self.generation.iloc[:, current_hour].copy()

        for bus_index in range(self.num_buses):
            if bus_index not in self.bus_index_name_mapping or bus_index not in updated_P_G.index:
                continue
            current_soc = self.battery_soc[bus_index]
            battery_capacity_mwh = self.battery_capacity[bus_index]
                
            P_ch = min(P_ch_max, (soc_max - current_soc) * battery_capacity_mwh / (self.eta_ch * self.E_max))
            P_dch = min(P_dch_max, (current_soc - soc_min) * battery_capacity_mwh * self.eta_dch / self.E_max)
                
            if current_hour % 2 == 0:
                current_soc += P_ch * self.eta_ch / battery_capacity_mwh
                updated_P_G[bus_index] += P_ch
            else:
                current_soc -= P_dch / (self.eta_dch * battery_capacity_mwh)
                updated_P_G[bus_index] -= P_dch
                
            self.battery_soc[bus_index] = min(max(current_soc, soc_min), soc_max)

        self.updated_P_G.append(updated_P_G)

    def add_loads(self):
        for bus_index in self.bus_index_name_mapping.keys():
            P_L = 0
            Q_L = 0
            pp.create_load(self.net, bus=bus_index, p_mw=P_L, q_mvar=Q_L, name=f"Load at {self.bus_index_name_mapping[bus_index]}")

    def update_hourly_changes(self, current_hour):
        for bus_idx in self.net.bus.index:
            zone = self.net.bus.at[bus_idx, 'zone']
            if zone == 'extgrid':
                continue  # Skip 'extgrid' buses

            # Find corresponding data for this bus if it exists
            if bus_idx in self.load_demand.index and bus_idx in self.generation.index:
                P_L = self.load_demand.iloc[bus_idx, current_hour]
                Q_L = P_L * 0.9  # Assuming a power factor for Q calculation

                P_G = self.generation.iloc[bus_idx, current_hour]
                Q_G = P_G * 0.9  # Assuming a power factor for Q calculation

                self.net.load.at[bus_idx, 'p_mw'] = P_L
                self.net.load.at[bus_idx, 'q_mvar'] = Q_L

                self.net.sgen.at[bus_idx, 'p_mw'] = P_G
                self.net.sgen.at[bus_idx, 'q_mvar'] = Q_G

                print(f"--bus id: {bus_idx} -P_L: {P_L}, Q_L: {Q_L}, P_G: {P_G}, Q_G: {Q_G}")

        self.update_battery_soc_and_pg(current_hour)

    def run_simulation(self):
        try:
            # Set the algorithm options
            pp.runpp(self.net, algorithm='nr', init='auto', tolerance=1e-6, max_iteration=20)
        except pp.powerflow.LoadflowNotConverged as e:
            print("Power flow did not converge:", e)
            return None

        P_line = self.net.res_line['p_from_mw'].values
        Q_line = self.net.res_line['q_from_mvar'].values
        V = self.net.res_bus['vm_pu'].values
        L_P = self.net.res_line['loading_percent'].values
        target = np.hstack([P_line, Q_line, V, L_P])[np.newaxis, :]
        P_G = self.net.sgen['p_mw'].values
        Q_G = self.net.sgen['q_mvar'].values
        P_L = self.net.load['p_mw'].values
        Q_L = self.net.load['q_mvar'].values
        inputs = np.hstack([P_G, Q_G, P_L, Q_L])[np.newaxis, :]
        return target, inputs

    def create_dataset(self, days=365, num_samples=1):
        hours_per_day = 24
        X, Y = [], []
        print(f"Starting dataset creation for {num_samples} samples, each with {days} days and {hours_per_day} hours per day.")
        for sample in range(num_samples):
            self.battery_soc = {bus_index: 0.5 for bus_index in range(self.num_buses)}  # Reset SOC for each sample
            for day in range(days):
                for hour in range(hours_per_day):
                    print(f"Sample {sample + 1}, Day {day + 1}, Hour {hour + 1}: Updating loads and generations.")
                    self.update_hourly_changes(hour)  # Pass the current hour to update conditions
                    target, inputs = self.run_simulation()
                    if target is not None and inputs is not None:
                        Y.append(target)
                        X.append(inputs)
                print(f"Completed simulations for Sample {sample + 1}, Day {day + 1}.")
        self.Y = np.vstack(Y)
        self.X = np.vstack(X)
        self.updated_P_G = np.vstack(self.updated_P_G)
        print("Dataset creation complete.")

    def save_dataset(self):
        print("Saving dataset to CSV")
        X_cols, Y_cols = self.get_columns()
        Y_df = pd.DataFrame(data=self.Y, columns=Y_cols)
        X_df = pd.DataFrame(data=self.X, columns=X_cols)
        updated_P_G_df = pd.DataFrame(data=self.updated_P_G, columns=[f"P_G{i}" for i in range(self.updated_P_G.shape[1])])
        try:
            Y_df.to_csv('./data/Y_PF.csv', index=False)
            X_df.to_csv('./data/X_PF.csv', index=False)
            updated_P_G_df.to_csv('./data/updated_P_G.csv', index=False)
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
        for i in range(len(self.net.res_line)):
            Y_cols.append(f"P_L_{i}")
        return X_cols, Y_cols

if __name__ == '__main__':
    config_data = {
        'demand_data': './data/load_demand.csv',
        'generation_data': './data/generation.csv'
    }
    sim = Simulation()
    sim.create_std_line()
    sim.initialize_network()
    sim.initialize_connection()
    sim.add_loads()
    sim.load_data(config_data['demand_data'], config_data['generation_data'])
    sim.create_dataset(days=365)
    sim.save_dataset()
