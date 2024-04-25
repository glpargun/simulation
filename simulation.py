import numpy as np
import pandapower as pp
import pandas as pd
import pandapower.plotting as plot
import pickle

from conf.bus_config import location_buses
from conf.connection_config import line_connections
from conf.line_config import line_data

class Simulation:
    def __init__(self, num_buses = 160, max_battery_capacity = 40):
        self.net = pp.create_empty_network()
        self.bus_index_name_mapping = {}
        self.simulation_inputs = {}

        self.p_load = np.random.uniform(10, 20) 
        self.q_load = np.random.uniform(5, 10)

        self.p_gen = np.random.uniform(self.p_load + 5, self.p_load + 10)
        self.q_gen = np.random.uniform(self.q_load + 5, self.q_load + 10)
        
        self.battery_soc = {bus_index: 0.5 for bus_index in range(num_buses)}
        self.num_buses = num_buses
        self.battery_capacity = {bus_index: max_battery_capacity for bus_index in range(num_buses)}

    def create_std_line(self):
        pp.create_std_type(self.net, line_data, name="Al/St 240/40 4-bundle 380.0", element='line')

    def initialize_network(self):

        for key, value in location_buses.items():
            # print(key, value[0])
            bus_index = pp.create_bus(self.net, 
                            vn_kv=380., 
                            name=f"Bus {key}", 
                            geodata = (value[0], value[1]),
                            zone = value[2],
                            max_vm_pu = np.inf, 
                            min_vm_pu = 0.0)
            self.bus_index_name_mapping[bus_index] = f"Bus {key}"
            
            if value[2] == 'extgrid':
                pp.create_ext_grid(self.net,
                                   bus=bus_index, name=key)
                self.bus_index_name_mapping[bus_index] = f"Bus {key}"
            else:                
                pp.create_sgen(self.net,
                               bus=bus_index,
                               p_mw=self.p_gen,
                               q_mvar=self.q_gen,
                               name=f"sgen {key}")
        # print(self.bus_index_name_mapping)

        return self.net
    
    def initialize_connection(self):
        for line in line_connections:
            from_bus = line["from_bus"]
            to_bus = line["to_bus"]
            length_km = line.get("length_km", 1)
            std_type = line["std_type"]
            num_parallel = max(line.get("num_parallel", 1), 1)
            
            from_bus_idx = pp.get_element_index(self.net, "bus", f"Bus {from_bus}")
            to_bus_idx = pp.get_element_index(self.net, "bus", f"Bus {to_bus}")

            # print(f"from: {from_bus_idx}, to: {to_bus_idx}")

            pp.create_line(self.net,
                           from_bus = from_bus_idx,
                           to_bus = to_bus_idx,
                           length_km=length_km,
                           std_type=std_type,
                           parallel=num_parallel)
        
        # print(self.net.line)
            
    def update_sgens_with_battery(self):
        charge_rate = 0.1
        discharge_rate = 0.1
        soc_min = 0.3
        soc_max = 0.7 
        
        for bus_index in range(self.num_buses):
            sgens_at_bus = self.net.sgen[self.net.sgen.bus == bus_index]
            
            if not sgens_at_bus.empty:
                sgen_id = sgens_at_bus.index[0]
                current_soc = self.battery_soc[bus_index]
                battery_capacity_mwh = self.battery_capacity[bus_index]

                if current_soc < soc_min:
                    charge_amount = charge_rate * battery_capacity_mwh
                    self.net.sgen.at[sgen_id, 'p_mw'] += charge_amount
                    current_soc += charge_amount / battery_capacity_mwh
                
                elif current_soc > soc_max:
                    discharge_amount = discharge_rate * battery_capacity_mwh
                    self.net.sgen.at[sgen_id, 'p_mw'] -= discharge_amount
                    current_soc -= discharge_amount / battery_capacity_mwh
                
                current_soc = min(max(current_soc, soc_min), soc_max) # be sure that soc is specified bounds.
                self.battery_soc[bus_index] = current_soc

                P_gen = self.net.sgen.at[sgen_id, 'p_mw']
                S_gen = np.sqrt(P_gen**2 + self.net.sgen.at[sgen_id, 'q_mvar']**2)
                delta = np.random.choice([-1, 1]) 
                Q_gen = delta * np.sqrt(max(0, S_gen**2 - P_gen**2))
                self.net.sgen.at[sgen_id, 'q_mvar'] = Q_gen

                self.simulation_inputs[bus_index] = {
                'p_gen': P_gen,
                'q_gen': Q_gen,
                'p_load': self.p_load,
                'q_load': self.q_load,
                'soc': current_soc,              
                'battery_capacity_mwh': battery_capacity_mwh  
                }

                # print("Keys in self.battery_soc:", self.battery_soc.keys())
                # print("Keys in self.simulation_inputs:", self.simulation_inputs.keys())

           
    def add_loads(self):
        for bus_index in self.bus_index_name_mapping.keys():
            pp.create_load(self.net, bus=bus_index, p_mw=self.p_load, q_mvar=self.q_load, name=f"Load at {self.bus_index_name_mapping[bus_index]}")
            self.simulation_inputs[bus_index] = {'p_gen': 0, 'q_gen': 0, 'p_load': self.p_load, 'q_load': self.q_load}

    def plot_network(self):
        plot.simple_plot(self.net, show_plot=True)

    def run_simulation(self):
        self.update_sgens_with_battery()
        pp.runpp(self.net) 

        P_line = self.net.res_line['p_from_mw'].tolist()
        Q_line = self.net.res_line['q_from_mvar'].tolist()
        V = self.net.res_bus['vm_pu'].tolist()

        y = [P_line, Q_line, V]

        print("\nSimulation output Y:")
        # print(y)
        # print(self.net.res_line)
        print("\nSimulation inputs X:")
        x = []
        for bus_id, inputs in self.simulation_inputs.items():
            bus_inputs = [inputs['p_gen'], inputs['q_gen'], inputs['p_load'], inputs['q_load']]
            x.append(bus_inputs)
        # print(x)
        print(self.simulation_inputs)
    


if __name__ == '__main__':
    sim = Simulation()
    sim.create_std_line()
    sim.initialize_network()
    sim.initialize_connection()
    # sim.plot_network()
    sim.add_loads()
    sim.update_sgens_with_battery()
    sim.run_simulation()