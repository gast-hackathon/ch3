
import numpy as np
import matplotlib.pyplot as plt

def get_full_columns(dataset, except_column, output_index):

	x_train = []
	y_train = []

	for col in range(dataset.shape[1]):

		if col != except_column:

			column = dataset[:, col]

			zeros = np.count_nonzero(column == 0)

			if zeros == 0:
				
				x = []
				y = 0

				for row in range(dataset.shape[0]):

					if row != output_index:
						x.append(dataset[row, col])
					else:
						y = dataset[row, col]
				
				x_train.append(x)
				y_train.append(y)
	
	return np.array(x_train), np.array(y_train)

def get_predict_column(dataset, column, output_index):

	x = []

	col = dataset[:, column]

	for i in range(col.shape[0]):

		if i != output_index:

			x.append(col[i])

	return np.array(x).reshape(1, len(x))

def normalize(x):
	mean = np.mean(x, axis=0)
	x = x - mean
	mx = np.max(x, axis=0)

	if not np.all(mx != 0):
		print("zeros", mx)
	if not np.isfinite(mx).all():
		print("infinite", mx)
	if not np.isfinite(x).all():
		print("infinite x", x)

	x = x / mx
	return x, (mean, mx)

def norm_apply(x, norm):
	mean = norm[0]
	mx = norm[1]
	return (x - mean) / mx

def denormalize(x, norm):
	mean = norm[0]
	mx = norm[1]
	return (x * mx) + mean

def assert_positive(dataset):
	for i in range(dataset.shape[0]):
		for j in range(dataset.shape[1]):
			assert dataset[i, j] > -0.001
			
