import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.plotting as plot
import pandapower.topology as top
from numba import jit

from conf.bus_config import location_buses
from conf.connection_config import line_connections
from conf.line_config import line_data
from conf.generators_config import generators

class PowerCalculation:
    def __init__(self, file_path, power_factor=0.8):
        self.data = pd.read_pickle(file_path) / 1000 # Convert kW to MW
        self.power_factor = power_factor
        self.arccos_pf = np.arccos(self.power_factor)
        self.tan_arccos_pf = np.tan(self.arccos_pf)

    def calculate_hourly_reactive_power(self):
        records_per_hour = 4
        hourly_data = (self.data.T
                       .groupby(np.arange(len(self.data.columns)) // records_per_hour)
                       .sum().T)
        # print(f"---hourly data shape: {hourly_data.shape}")

        reactive_power = hourly_data * self.tan_arccos_pf
        # print(f"---reactive_power data shape: {reactive_power.shape}")
        return reactive_power.round(2)

    @staticmethod
    def load_and_resample_generator_data(generator_path):
        generator_data = pd.read_pickle(generator_path) / 1000  # Convert kW to MW
        records_per_hour = 4
        hourly_data = (generator_data.T
                       .groupby(np.arange(len(generator_data.columns)) // records_per_hour)
                       .sum().T)
        return hourly_data.round(2)


class SimulationDataPreparer:
    @staticmethod
    def create_new_std_line(net, line_data):
        """Adds a new standard line type to the Pandapower network."""
        pp.create_std_type(net, line_data, name="Al/St 240/40 4-bundle 380.0", element='line')

    @staticmethod
    def prepare_bus_location_data(bus_locations, net):
        """Prepares bus location data for review."""
        for bus_id, (lon, lat, region) in bus_locations.items():
            if region == 'extgrid':
                pp.create_ext_grid(net, bus=pp.create_bus(net, vn_kv=380, name=f"Bus {bus_id}", geodata=(lon, lat), zone=region, max_vm_pu=np.inf, min_vm_pu=0.0))
            else:
                pp.create_bus(net, vn_kv=380, name=f"Bus {bus_id}", geodata=(lon, lat), zone=region, max_vm_pu=np.inf, min_vm_pu=0.0)

    @staticmethod
    def prepare_line_connection_data(line_connections):
        """Converts line connection information into a pandas DataFrame."""
        line_data = pd.DataFrame(line_connections)
        return line_data

    @staticmethod
    def create_lines(net, line_connections):
        for connection in line_connections:
            from_bus = connection["from_bus"]
            to_bus = connection["to_bus"]
            length_km = connection.get("length_km", 1)
            std_type = connection["std_type"]
            num_parallel = max(connection.get("num_parallel", 1), 1)

            from_bus_idx = pp.get_element_index(net, "bus", f"Bus {from_bus}")
            to_bus_idx = pp.get_element_index(net, "bus", f"Bus {to_bus}")

            pp.create_line(net, from_bus=from_bus_idx, to_bus=to_bus_idx, length_km=length_km, std_type=std_type, parallel=num_parallel)

    @staticmethod
    def add_sgens_from_generators(net, generator_reactive, bus_name_to_index_map, location_buses, generator_data_hourly, time_point):
        """
        Adds static generators (sgens) to the network based on provided generator data.
        Uses reactive power data from generator_reactive calculations.

        :param net: The Pandapower network.
        :param generator_reactive: DataFrame containing reactive power data for generators.
        :param bus_name_to_index_map: Mapping from bus names to Pandapower bus indices.
        :param location_buses: Dictionary containing bus location information.
        :param generator_data_hourly: Hourly generator data after resampling.
        """
        # Counting buses in each region
        region_counts = {}
        for _, (_, _, region) in location_buses.items():
            if region != 'extgrid':  # Exclude buses with 'extgrid' region
                region_counts[region] = region_counts.get(region, 0) + 1
        # print(f"---- region: {region_counts}")

        for region_idx in range(1, 78):
            total_buses_in_region = region_counts.get(region_idx, 0)
            if total_buses_in_region == 0:
                print(f"No buses found in region {region_idx}. Skipping...")
                continue

            p_mw = generator_data_hourly.iloc[region_idx, time_point] / total_buses_in_region  # Active power in MW
            q_mvar = generator_reactive.iloc[region_idx, time_point] / total_buses_in_region  # Reactive power in MVAr
            # print(f"SGEN: Region: {region_idx}, shape: {generator_data_hourly.shape}, active: {p_mw}, reactive: {q_mvar}")
            for bus_name, (_, _, bus_region) in location_buses.items():
                if bus_region == region_idx and bus_region != 'extgrid':
                    bus_index = bus_name_to_index_map.get(bus_name)
                    if bus_index is not None:
                        pp.create_sgen(net, bus=bus_index, p_mw=p_mw, q_mvar=q_mvar, name=f"Sgen_{bus_name}")
                    else:
                        print(f"Bus {bus_name} not found in bus_name_to_index_map.")

    @staticmethod
    def add_loads_to_network(net, load_active, load_reactive, bus_name_to_index_map, location_buses, time_point):
        """
        Adds loads to the network based on provided active and reactive power data, distributed by region.

        :param net: The Pandapower network.
        :param load_active: DataFrame containing active power data for loads.
        :param load_reactive: DataFrame containing reactive power data for loads.
        :param bus_name_to_index_map: Mapping from bus names to Pandapower bus indices.
        :param location_buses: Dictionary containing bus location information, including regions.
        """
        # Counting buses in each region
        region_counts = {}
        for _, (_, _, region) in location_buses.items():
            if region != 'extgrid':  # Exclude 'extgrid' region for loading
                region_counts[region] = region_counts.get(region, 0) + 1

        # Iterating over all the regions
        for region_idx in range(1, 78):  # Assuming region indices range from 1 to 78
            total_buses_in_region = region_counts.get(region_idx, 0)
            if total_buses_in_region == 0:
                print(f"No buses found in region {region_idx}. Skipping...")
                continue

            # Extracting single values for p_mw and q_mvar for the region
            p_mw = load_active.iloc[region_idx, time_point] / total_buses_in_region  # Active power in MW
            q_mvar = load_reactive.iloc[region_idx, time_point] / total_buses_in_region  # Reactive power in MVAr

            # Create a load on each bus within the region
            for bus_name, (_, _, bus_region) in location_buses.items():
                if bus_region == region_idx and bus_region != 'extgrid':
                    bus_index = bus_name_to_index_map.get(bus_name)
                    if bus_index is not None:
                        pp.create_load(net, bus=bus_index, p_mw=p_mw, q_mvar=q_mvar, name=f"Load_{bus_name}")
                        # print(f"LOAD: region: {bus_region},bus id: {bus_index}, active: {p_mw}, reactive: {q_mvar}")
                    else:
                        print(f"Bus {bus_name} not found in bus_name_to_index_map.")

    @staticmethod
    def info_of_lines_and_export_data(net):
        """Calculate the loading percentage for each line in the network and export data."""
        
        # Extract relevant information
        loading_info = net.res_line[['loading_percent', 'p_from_mw', 'q_from_mvar']]
        
        # Filter lines with loading percentage greater than 100 and display info
        for line_idx, info in loading_info.iterrows():
            loading_percent = info['loading_percent']
            if loading_percent > 100:
                print(f"Line {line_idx}: Loading Percentage: {loading_percent:.2f}%, P from bus: {info['p_from_mw']:.2f} MW, Q from bus: {info['q_from_mvar']:.2f} MVAr")
        
        # Export data to CSV files
        loading_info[['loading_percent']].to_csv('./data/export/loading_percent.csv', index=True, header=True)
        loading_info[['p_from_mw']].to_csv('./data/export/p_from_mw.csv', index=True, header=True)
        loading_info[['q_from_mvar']].to_csv('./data/export/q_from_mvar.csv', index=True, header=True)

    def print_net_info(net):
        print("Network Information:")
        print("-------------------")
        print(f"Number of Buses: {len(net.bus)}")
        print(f"Number of Lines: {len(net.line)}")
        print(f"Number of Transformers: {len(net.trafo)}")
        print(f"Number of Loads: {len(net.load)}")
        print(f"Number of Static Generators (sgen): {len(net.sgen)}")
        print(f"Number of External Grids: {len(net.ext_grid)}")
        print(f"Number of Switches: {len(net.switch)}")


class pre_check_simulation:

    @staticmethod
    def advanced_pre_simulation_checks(net):
        print("Performing advanced pre-simulation checks...")

        # Exclude external grid buses from certain checks
        ext_grid_buses = set(net.ext_grid.bus.values)

        # Check for attached loads or sgens, ignoring external grid buses
        for bus in net.bus.index:
            if bus in ext_grid_buses:
                continue  # Skip checks for external grid buses
            has_load = net.load[net.load.bus == bus].shape[0] > 0
            has_sgen = net.sgen[net.sgen.bus == bus].shape[0] > 0
            if not (has_load or has_sgen):
                print(f"Warning: Bus {bus} does not have any load or static generator attached.")

    @staticmethod
    def balance_generation_to_load(net):
        print("Balancing total generation with total load to mitigate significant power imbalance.")
        
        total_load_p = net.load.p_mw.sum()
        total_generation_p = net.sgen.p_mw.sum() + net.gen.p_mw.sum()

        # Check if 'q_mvar' column exists in load DataFrame
        if 'q_mvar' in net.load.columns:
            total_load_q = net.load.q_mvar.sum()
        else:
            print("Warning: 'q_mvar' column does not exist in net.load. Assuming zero reactive power for loads.")
            total_load_q = 0

        # Check if 'q_mvar' column exists in sgen DataFrame
        if 'q_mvar' in net.res_sgen.columns:
            total_generation_q_sgen = net.sgen.q_mvar.sum()
        else:
            print("Warning: 'q_mvar' column does not exist in net.sgen. Assuming zero reactive power for static generators.")
            total_generation_q_sgen = 0

        total_generation_q = total_generation_q_sgen

        # Balancing active power
        desired_total_generation_p = total_load_p * 1.1  # 10% reserve margin
        scaling_factor_p = desired_total_generation_p / total_generation_p if total_generation_p != 0 else 1

        # Balancing reactive power
        desired_total_generation_q = total_load_q * 1.1  # Assuming a 10% reserve margin for reactive power as well
        scaling_factor_q = desired_total_generation_q / total_generation_q if total_generation_q != 0 else 1

        net.sgen.p_mw *= scaling_factor_p
        net.gen.p_mw *= scaling_factor_p

        if 'q_mvar' in net.sgen.columns:
            net.sgen.q_mvar *= scaling_factor_q
        if 'q_mvar' in net.gen.columns:
            net.gen.q_mvar *= scaling_factor_q

        print(f"After balancing, Total Generation P: {net.sgen.p_mw.sum() + net.gen.p_mw.sum()}, Total Load P: {net.load.p_mw.sum()}")
        print(f"After balancing, Total Generation Q: {total_generation_q_sgen}, Total Load Q: {total_load_q}")

        # Check for reactive power imbalance
        if abs(total_generation_q - total_load_q) > 10:  # Define threshold based on your network
            print("Significant reactive power imbalance detected.")

    @staticmethod
    def check_network_connectivity(net):
        print("Performing connectivity check to identify isolated or unsupplied buses.")
        unsupplied_buses = top.unsupplied_buses(net)
        if len(unsupplied_buses) > 0:
            print(f"Warning: Identified isolated or unsupplied buses - {unsupplied_buses}.")
        else:
            print("Connectivity check passed: No isolated or unsupplied buses identified.")

    @staticmethod
    def ensure_correct_voltage_levels(net):
        for _, row in net.bus.iterrows():
            if row.vn_kv < 0.1 or row.vn_kv > 380:
                print(f"Bus {row.name} has an unusual voltage level: {row.vn_kv}kV")

    @staticmethod
    def check_reactive_power_balance(net):
        q_generation_capacity = sum([gen.q_mvar for gen in net.sgen.itertuples()]) + sum([gen.q_mvar for gen in net.gen.itertuples()])
        q_load_demand = net.load.q_mvar.sum()
        if abs(q_generation_capacity - q_load_demand) > 10:  # Define threshold based on your network
            print("Significant reactive power imbalance detected.")

    @staticmethod
    def try_different_algorithms(net):
        algorithms = ['gs', 'bfsw']
        for alg in algorithms:
            try:
                pp.runpp(net, algorithm=alg)
                print(f"Power flow successful with {alg} algorithm.")
                break
            except Exception as e:
                print(f"Power flow with {alg} algorithm failed due to: {e}")

    @staticmethod
    def check_and_mitigate_overloads(net):
        """
        Checks for overloads in the network and attempts to mitigate them by
        adjusting generation or shedding load.
        """
        print("Checking for overloads...")
        pp.runpp(net, algorithm='nr')  # Running initial power flow to detect overloads

        # Identifying overloaded lines
        overloaded_lines = net.res_line[net.res_line.loading_percent > 100].index.tolist()
        overloaded_trafos = net.res_trafo[net.res_trafo.loading_percent > 100].index.tolist()

        if overloaded_lines or overloaded_trafos:
            print(f"Overloaded lines: {overloaded_lines}, Overloaded transformers: {overloaded_trafos}")
            # Implement mitigation strategy here, for example:
            # - Adjusting generation
            # - Load shedding
            # - Re-routing power (if applicable)
            # This is a placeholder for your specific logic
            print("Implementing mitigation strategy...")
            # Example mitigation strategy: Reducing load proportionally
            net.load.loc[:, 'p_mw'] = net.load.loc[:, 'p_mw'] * 0.9
            print("Re-running power flow after mitigation...")
            pp.runpp(net, algorithm='nr')
        else:
            print("No overloads detected.")

def main():
    net = pp.create_empty_network()

    # data preparing
    generator_path = './data/generation.pickle'
    load_path = './data/demand.pickle'
    generator_reactive = PowerCalculation(generator_path)
    load_reactive = PowerCalculation(load_path, power_factor=0.9)
    generator_reactive_power = generator_reactive.calculate_hourly_reactive_power()
    load_reactive_power = load_reactive.calculate_hourly_reactive_power()
    generator_data_hourly = PowerCalculation.load_and_resample_generator_data(generator_path)

    # Mapping buses:
    bus_name_to_index_map = {f"{bus_id}": index for index, bus_id in enumerate(location_buses.keys()) if not bus_id.startswith(('AT', 'CH', 'CZ', 'LU', 'NL', 'FR', 'PL', 'DK'))}
    # print(f"----bus_name_to_index: {bus_name_to_index_map.keys()}")  # Print the keys

    # print(f"Shape of generator_data_hourly: {generator_data_hourly.shape}")
    # print(f"Sample reactive power calculation: {generator_reactive_power.head()}")


    # setup
    SimulationDataPreparer.create_new_std_line(net, line_data)
    SimulationDataPreparer.prepare_bus_location_data(location_buses, net)
    SimulationDataPreparer.create_lines(net, line_connections)
    SimulationDataPreparer.add_sgens_from_generators(net, generator_reactive_power, bus_name_to_index_map, location_buses, generator_data_hourly, 5000)
    SimulationDataPreparer.add_loads_to_network(net, load_reactive_power, load_reactive_power, bus_name_to_index_map, location_buses, 5000)

    # plotting lines and buses with real geo data from pysa-eu
    # try:
    #     plot.simple_plot(net, show_plot=True)
    # except ImportError:
    #     print("Matplotlib not installed. Please install it for plotting.")


    # Perform advanced pre-simulation checks
    pre_check_simulation.advanced_pre_simulation_checks(net)
    pre_check_simulation.check_network_connectivity(net)
    pre_check_simulation.balance_generation_to_load(net)
    pre_check_simulation.ensure_correct_voltage_levels(net)
    pre_check_simulation.check_reactive_power_balance(net)
    pre_check_simulation.try_different_algorithms(net)
    pre_check_simulation.check_and_mitigate_overloads(net)

    # results
    SimulationDataPreparer.info_of_lines_and_export_data(net)

    SimulationDataPreparer.print_net_info(net)

    # pp.diagnostic(net)

    pp.runpp(net)

if __name__ == "__main__":
    main()
