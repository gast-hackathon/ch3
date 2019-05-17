
#%% Import libs
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.utils import normalize

import importlib
from livelossplot.keras import PlotLossesCallback

#%% Import locals
import load
import utils
importlib.reload(load)
importlib.reload(utils)

#%% Load dataset

x_train, y_train = load.load_all()

samples = y_train.shape[0]

x_train = normalize(x_train, axis=1)
y_train = normalize(y_train, axis=0).reshape((samples,))

print("x_train", x_train.shape)
print("y_train", y_train.shape)

#%% Build model

model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1]))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

#%% Train model

history = model.fit(x_train, y_train, 
	epochs=200,
	validation_split=0.35,
	verbose=0)

utils.plot_loss(history)
