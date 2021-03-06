
import time
import sys
import numpy as np
import xlrd
from xlrd.sheet import Sheet

import utils
import models

COLUMNS = 24
FEATURES = 5
countries = [
	'AT', 'CZ', 'DE', 'GR', 'HT', 'MT', 'NL', 'PL', 'SK'
]


ROWS = FEATURES+3
JOINT_COLUMNS = COLUMNS * len(countries)

dataset = np.zeros((1,))

def _col_x(col):

	if (col[2] + col[4]) > 0:
		rej_rate = ((col[1] * col[2]) + (col[3] * col[4])) / (col[2] + col[4])
	else:
		rej_rate = 0

	accepted_applications = (col[2] + col[4]) * (1 - rej_rate)

	if accepted_applications > 0:
		fraud_rate = col[11] / accepted_applications
	else:
		fraud_rate = 0

	return [
		col[2] + col[4],  # 0. Applications
		rej_rate,         # 1. Rejection rate
		col[9],           # 2. Churn rate
		col[10],          # 3. Forced disconnection
		col[14],          # 4. Average amount per fraud case
		fraud_rate,       # 5. Fraud rate
		col[14],          # 6. Fraud revenue
		col[7],           # 7. Never payer ratio
	]

def _load_dataset():

	global dataset

	if dataset.shape[0] > 1:
		return dataset

	dataset = np.zeros((ROWS, JOINT_COLUMNS))

	wb = xlrd.open_workbook("DTSE_Hackathon_C3_Credit_Forecast_data.xlsx")

	for i in range(len(countries)):

		sheet = wb.sheet_by_name(countries[i])

		for col in range(sheet.ncols - 2):

			col_vals = _col_x( sheet.col_values(col + 2) )
			col_vals = [x if x and x > 0 else 0 for x in col_vals]

			dataset[:, col + COLUMNS*i] = col_vals
	
	train_models = {}

	for row in range(ROWS):
		
		for col in range(JOINT_COLUMNS):

			if dataset[row, col] == 0:
				x_train, y_train = utils.get_full_columns(dataset, col, row)
				x_pred = utils.get_predict_column(dataset, col, row)

				if row not in train_models:
					model = models.LinearReg(x_train, y_train)
					model.set_epochs(200)
					model.set_non_neg()
					model.train()
					train_models[row] = model

				prediction = train_models[row].predict(x_pred)

				dataset[row, col] = prediction

	#utils.assert_positive(dataset)

	return dataset

def _load_country(country_code, with_index=False):

	ds = _load_dataset()
	ci = countries.index(country_code)

	x_train = []
	y_train = []

	for col in range(COLUMNS):

		col_values = ds[:, col + COLUMNS*ci]
		x = col_values[0:FEATURES].tolist()
		y = col_values[FEATURES]

		if with_index:
			for i in range(len(countries)):
				if i==ci:
					x.insert(0, 1)
				else:
					x.insert(0, 0)

		x_train.append(x)
		y_train.append(y)

	x_train, y_train = np.array(x_train), np.array(y_train)

	return x_train, y_train

def _load_all_countries():

	x_train = []
	y_train = []

	for i in range(len(countries)):

		x, y = _load_country(countries[i], with_index=True)
		
		x_train.append(x)
		y_train.append(y)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_train = x_train.reshape( (x_train.shape[0] * x_train.shape[1], x_train.shape[2]) )
	y_train = y_train.reshape( y_train.shape[0] * y_train.shape[1] )

	return x_train, y_train

def set_full_dataset(full_dataset):
	global dataset
	dataset = full_dataset

def country_average_payer_revenue(country):
	ds = _load_dataset()
	ci = countries.index(country)
	
	vals = []	
	
	for col in range(COLUMNS):
		col_values = ds[:, col + COLUMNS*ci]
		avg_rev = col_values[FEATURES+1]
		vals.append(avg_rev)

	return np.average(vals)

class Dataset:
		
	def __init__(self, country=False, with_index=False):

		self.country = country

		if country:
			self.x_train, self.y_train = _load_country(country, with_index)
		else:
			self.x_train, self.y_train = _load_all_countries()


	def country_average(self):

		avg_x = np.average(self.x_train, axis=0)
		avg_y = np.average(self.y_train)
		
		return avg_x.reshape((1, len(avg_x))), avg_y

	def print_shapes(self):
		print("x_train", self.x_train.shape)
		print("y_train", self.y_train.shape)

	def to_model(self):
		return models.LinearReg(self.x_train, self.y_train)

	def get_full_dataset(self):
		global dataset
		return dataset

def load_dataset(country=False, with_index=False):

	return Dataset(country, with_index)

if __name__ == "__main__":
	#ds = Dataset()
	
	for country in countries:

		ds = Dataset(country)

		_, y = ds.country_average()
		y = round(y*100, 2)

		print(country, y)
		print(ds.y_train)

