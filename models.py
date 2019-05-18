
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.regularizers import l1, l2, l1_l2
from keras.constraints import NonNeg
from keras.callbacks import ModelCheckpoint

import importlib
from livelossplot.keras import PlotLossesCallback

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

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
		self.use_pca = False
		self.x_train_pca = []
		self.non_neg = False
		self.checkpoint = ''

		self.samples = y_train.shape[0]

		self.x_train, self.x_norm = utils.normalize(x_train)
		self.y_train, self.y_norm = utils.normalize(y_train)

	def set_pca(self, n_features=6):
		self.use_pca = True
		self.pca = PCA(n_components=n_features)
		self.x_train_pca = self.pca.fit_transform(self.x_train)

	def set_epochs(self, epochs):
		self.epochs = epochs

	def set_non_neg(self):
		self.non_neg = True

	def set_checkpoint(self, filepath):
		self.checkpoint = filepath

	def train(self):

		if self.use_pca:
			x = self.x_train_pca
		else:
			x = self.x_train

		input_dim = x.shape[1]

		hidden = Dense(6, input_dim=input_dim, kernel_regularizer=l1(0.01))

		if self.non_neg:
			output = Dense(1, input_dim=input_dim, kernel_regularizer=l1(0.01), kernel_constraint=NonNeg())
		else:
			output = Dense(1, input_dim=input_dim, kernel_regularizer=l1(0.01))

		model = Sequential()

		# Hidden layer
		model.add(hidden)
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.4))

		# Output layer
		model.add(output)
		model.add(Activation('relu'))

		model.compile(loss='mean_squared_error', optimizer='adam')

		self.model = model

		callbacks = []

		if self.checkpoint:
			callbacks.append(ModelCheckpoint(self.checkpoint, monitor='loss'))

		self.history = self.model.fit(
			x, self.y_train, 
			epochs=self.epochs,
			validation_split=0.15,
			verbose=0,
			callbacks=callbacks)

		return self.history

	def plot_history(self, show=True):
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		if show:
			plt.show()
	
	def predict(self, x):
		x = utils.norm_apply(x, self.x_norm)
		if self.use_pca:
			x = self.pca.transform(x)
		y = self.model.predict(x)
		return utils.denormalize(y, self.y_norm)[0][0]

if __name__ == "__main__":
	x_train, y_train = _load_data(country="GR")
	print(y_train)
