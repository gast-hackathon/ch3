
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.regularizers import l1, l2, l1_l2

import importlib
from livelossplot.keras import PlotLossesCallback

import data
import utils
importlib.reload(data)
importlib.reload(utils)

def _load_data(country=False):

	ds = data.Dataset(country)

	return ds.x_train, ds.y_train

class LinearReg:

	def __init__(self, x_train, y_train):

		self.epochs = 200

		self.samples = y_train.shape[0]

		self.x_train, self.x_norm = utils.normalize(x_train)
		self.y_train, self.y_norm = utils.normalize(y_train)

		model = Sequential()
		model.add(Dense(8, input_dim=x_train.shape[1], kernel_regularizer=l2(0.01)))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))

		model.add(Dense(1, activation='tanh', kernel_regularizer=l2(0.01)))

		model.compile(loss='mean_squared_error', optimizer='adam')

		self.model = model

	def set_epochs(self, epochs):
		self.epochs = epochs

	def train(self):

		self.history = self.model.fit(
			self.x_train, self.y_train, 
			epochs=self.epochs,
			validation_split=0.35,
			verbose=0)

		return self.history

	def plot_history(self):
		utils.plot_loss(self.history)
	
	def predict(self, x):
		x = utils.norm_apply(x, self.x_norm)
		y = self.model.predict(x)
		return utils.denormalize(y, self.y_norm)[0][0]

if __name__ == "__main__":
	x_train, y_train = _load_data(country="GR")
	print(y_train)
