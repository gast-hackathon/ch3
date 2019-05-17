
#%% Import libs
import time
import importlib


#%% Import locals
import utils
importlib.reload(utils)


#%% Load dataset
import data
importlib.reload(data)

start = time.time()

all_countries_dataset = data.load_dataset()
full_dataset = all_countries_dataset.get_full_dataset()

print("preprocessing:", (time.time()-start), "sec")

#%% Train
import models
importlib.reload(models)

start = time.time()

model = all_countries_dataset.to_model()
model.set_epochs(300)
model.train()

print("training:", (time.time()-start), "sec")

model.plot_history()

#%% Exhaustive search
import algo
importlib.reload(algo)

algo.exhaustive_search(model, full_dataset, "CZ")
