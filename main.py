
#%% Import libs
import time
import importlib

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#%% Import locals
import utils
importlib.reload(utils)


#%% Load dataset
import data
importlib.reload(data)

start = time.time()

dataset = data.load_dataset()
full_dataset = dataset.get_full_dataset()

print("preprocessing:", (time.time()-start), "sec")

print("train_data", dataset.x_train.shape)

#%% Train
import models
importlib.reload(models)

start = time.time()

model = dataset.to_model()
model.set_epochs(300)
model.set_pca(4)
model.train()

print("training:", (time.time()-start), "sec")

#model.plot_history()

#%% Exhaustive search
import algo
importlib.reload(algo)

for country in data.countries:
	algo.exhaustive_search(model, full_dataset, country)

algo.plot()
