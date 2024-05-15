import pandas as pd

# Load the original pickle files
load_demand = pd.read_pickle('./data/demand.pickle')
generation = pd.read_pickle('./data/generation.pickle')

# Convert to CSV
load_demand.to_csv('./data/load_demand.csv', index=False)
generation.to_csv('./data/generation.csv', index=False)