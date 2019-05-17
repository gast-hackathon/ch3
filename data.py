
import sys
import numpy as np
import xlrd
from xlrd.sheet import Sheet

import utils
import models

COLUMNS = 24
ROWS = 10
countries = [
	'AT', 'CZ', 'DE', 'GR', 'HT', 'MT', 'NL', 'PL', 'SK'
]
JOINT_COLUMNS = COLUMNS * len(countries)

dataset = np.zeros((1,))

def _col_x(col):

	return [
		col[2] + col[4],  # 0. Applications
		col[22],          # 1. Rejections
		col[8],           # 2. Never payer
		col[9],           # 3. Churn rate
		col[10],          # 4. Forced disconnection
		col[14],          # 5. Average amount per fraud case
		col[15],          # 6. Subscribers blocked - fraud
		col[16],          # 7. Subscribers blocked - all
		col[17],          # 8. Balance blocked - credit monitoring
		col[11],          # 9. Fraud count
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

			col_vals = _col_x( sheet.col_values(col + 2)[1:] )
			col_vals = [x if x else 0 for x in col_vals]

			dataset[:, col] = col_vals
	
	train_models = {}

	for row in range(ROWS):
		
		for col in range(JOINT_COLUMNS):

			if dataset[row, col] == 0:
				x_train, y_train = utils.get_full_columns(dataset, col, row)
				x_pred = utils.get_predict_column(dataset, col, row)

				if row not in train_models:
					model = models.LinearReg(x_train, y_train)
					model.set_epochs(200)
					model.train()
					train_models[row] = model

				prediction = train_models[row].predict(x_pred)

				dataset[row, col] = prediction

	utils.assert_no_zeros(dataset)

	return dataset

def _load_country(country_code, country_index=-1):

	ds = _load_dataset()
	ci = countries.index(country_code)

	x_train = []
	y_train = []

	for col in range(COLUMNS):

		col_values = ds[:, col + COLUMNS*ci]
		x = col_values[:-1].tolist()
		y = col_values[9]

		if country_index > -1:
			for i in range(len(countries)):
				if i==country_index:
					x.insert(0, 1)
				else:
					x.insert(0, 0)

		x_train.append(x)
		y_train.append(y)

	return x_train, y_train

def _load_all_countries():

	x_train = []
	y_train = []

	for i in range(len(countries)):

		x, y = _load_country(countries[i], country_index=i)
		
		x_train.append(x)
		y_train.append(y)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_train = x_train.reshape( (x_train.shape[0] * x_train.shape[1], x_train.shape[2]) )
	y_train = y_train.reshape( y_train.shape[0] * y_train.shape[1] )

	return x_train, y_train

class Dataset:
		
	def __init__(self, country=False):
		
		if country:
			self.x_train, self.y_train = _load_country(country)
		else:
			self.x_train, self.y_train = _load_all_countries()


	def country_average(self):

		avg = np.average(self.x_train, axis=0)
		
		return avg

	def print_shapes(self):
		print("x_train", self.x_train.shape)
		print("y_train", self.y_train.shape)

if __name__ == "__main__":
	#ds = Dataset()
	
	_load_dataset()


