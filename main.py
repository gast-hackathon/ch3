
#%% Import libs
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.utils import normalize

import importlib
from livelossplot.keras import PlotLossesCallback

#%% Import locals
import data
import utils
import models
importlib.reload(data)
importlib.reload(utils)
importlib.reload(models)


#%% Train

model = models.load_model(epochs=500)

model.plot_history()
