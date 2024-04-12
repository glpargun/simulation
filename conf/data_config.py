import pandas as pd
import numpy as np

start_date = '2019-01-01 00:00:00'
periods = 35040
date_range = pd.date_range(start=start_date, periods=periods, freq='15min')

# Load demand and generation data
load = pd.read_pickle('./data/demand.pickle').T
generator = pd.read_pickle('./data/generation.pickle').T

load.index = date_range
generator.index = date_range

load = load.resample('h').sum()
generator = generator.resample('h').sum()

assert load.shape[0] == 8760
assert generator.shape[0] == 8760

load = np.round(load.values.reshape(365, 24, 78) / 1000, 2)
generator = np.round(generator.values.reshape(365, 24, 78) / 1000, 2)

print(f"---load shape: {load.shape}, ---generator shape: {generator.shape}")
print(f"--- load: {load}, --- generator: {generator}")