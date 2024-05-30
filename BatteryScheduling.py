import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from simulation import Simulation

from conf.connection_config import line_connections

class BatteryScheduler:
    def __init__(self, num_buses=160, learning_rate=0.01, iterations=100, soc_min=0.3, soc_max=0.7, alpha=0.01):
        self.num_buses = num_buses
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.alpha = alpha  # Regularization parameter
        self.threshold = 80  # Line loading threshold (e.g., 80%)

    def objective_function(self, SOC, dSOC, line_loadings):
        penalty_soc_bounds = np.sum((SOC < self.soc_min) + (SOC > self.soc_max))
        deviation_from_target = np.sum((SOC - 0.5)**2)
        smoothness_penalty = np.sum((SOC[1:] - SOC[:-1] - dSOC[:-1])**2)
        line_loading_penalty = np.sum((line_loadings - self.threshold)**2)
        return penalty_soc_bounds + deviation_from_target + smoothness_penalty + self.alpha * line_loading_penalty

    def optimize_battery_schedule(self, line_loadings, initial_SOC=0.5):
        SOC = np.full(24, initial_SOC)
        dSOC = np.zeros(23)  # dSOC should have length 23 for hourly differences in a day

        for _ in range(self.iterations):
            grad_SOC = 2 * (SOC - 0.5)
            grad_dSOC = 2 * (line_loadings - self.threshold)

            SOC[:-1] -= self.learning_rate * grad_SOC[:-1]
            dSOC -= self.learning_rate * grad_dSOC[:-1]

            SOC = np.clip(SOC, self.soc_min, self.soc_max)
            SOC[1:] = SOC[:-1] + dSOC  # Ensure SOC transition constraint

        return SOC, dSOC

def load_line_loading_data(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_daily_battery_status(SOC, dSOC, line_loadings, day, bus_index_name_mapping, line_connections):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(SOC, label='State of Charge (SOC)')
    plt.xlabel('Hour')
    plt.ylabel('SOC')
    plt.title(f'Battery SOC for Day {day}')
    plt.legend()

    for i, (soc, dsoc) in enumerate(zip(SOC, np.concatenate(([0], dSOC)))):
        if dsoc > 0:
            plt.annotate('Charging', xy=(i, soc), xytext=(i, soc + 0.05), arrowprops=dict(facecolor='green', shrink=0.05))
        elif dsoc < 0:
            plt.annotate('Discharging', xy=(i, soc), xytext=(i, soc - 0.05), arrowprops=dict(facecolor='red', shrink=0.05))

    plt.subplot(3, 1, 2)
    plt.plot(line_loadings, label='Line Loadings')
    plt.axhline(y=80, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Hour')
    plt.ylabel('Loading %')
    plt.title(f'Line Loadings for Day {day}')
    plt.legend()

    overloaded_lines = np.where(line_loadings > 80)[0]
    underloaded_lines = np.where(line_loadings < 20)[0]

    for line in overloaded_lines:
        plt.annotate('Overloaded', xy=(line, line_loadings[line]), xytext=(line, line_loadings[line] + 5), arrowprops=dict(facecolor='red', shrink=0.05))
    
    for line in underloaded_lines:
        plt.annotate('Underloaded', xy=(line, line_loadings[line]), xytext=(line, line_loadings[line] - 5), arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.subplot(3, 1, 3)
    plt.bar(range(len(bus_index_name_mapping)), [SOC[-1]]*len(bus_index_name_mapping))
    plt.xlabel('Bus')
    plt.ylabel('SOC')
    plt.title(f'Battery SOC at Buses for Day {day}')
    plt.xticks(range(len(bus_index_name_mapping)), list(bus_index_name_mapping.keys()), rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_annual_battery_status(scheduler, line_loadings, bus_index_name_mapping, line_connections):
    for day in range(1, 366):
        line_loadings_example = line_loadings.iloc[day - 1].values[:24]  # Assuming each row corresponds to a day and only 24 values per day

        optimized_SOC, optimized_dSOC = scheduler.optimize_battery_schedule(line_loadings_example)
        
        # Plotting the results
        plot_daily_battery_status(optimized_SOC, optimized_dSOC, line_loadings_example, day, bus_index_name_mapping, line_connections)

if __name__ == '__main__':
    line_loadings = load_line_loading_data('./data/Y_PF.csv')
    
    scheduler = BatteryScheduler()

    # Load simulation to get the bus index name mapping
    sim = Simulation()
    sim.create_std_line()
    sim.initialize_network()
    sim.initialize_connection()
    sim.add_loads()
    bus_index_name_mapping = sim.bus_index_name_mapping
    line_connections = line_connections  # Assuming this variable is imported from the simulation configuration

    # Visualize the battery status for the entire year
    plot_annual_battery_status(scheduler, line_loadings, bus_index_name_mapping, line_connections)
