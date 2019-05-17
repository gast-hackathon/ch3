
import numpy as np
import xlrd
from xlrd.sheet import Sheet

countries = [
	'AT', 'CZ', 'DE', 'GR', 'HT', 'MT', 'NL', 'PL', 'SK'
]

def col_x(col):

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
	]


def load_country(country_code, country_index=-1):

	wb = xlrd.open_workbook("DTSE_Hackathon_C3_Credit_Forecast_data.xlsx")

	sheet = wb.sheet_by_name(country_code)

	x_train = []
	y_train = []

	for col in range(sheet.ncols - 2):
		col_values = sheet.col_values(col + 2)
		x = col_x(col_values)
		y = col_values[11]

		if country_index > -1:
			x.insert(0, country_index)

		x_train.append(x)
		y_train.append(y)
	
	return np.array(x_train), np.array(y_train)

def load_all():

	x_train = []
	y_train = []

	for i in range(len(countries)):

		x, y = load_country(countries[i], country_index=i)
		
		x_train.append(x)
		y_train.append(y)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_train = x_train.reshape( (x_train.shape[0] * x_train.shape[1], x_train.shape[2]) )
	y_train = y_train.reshape( y_train.shape[0] * y_train.shape[1] )

	return x_train, y_train

if __name__ == "__main__":
	x, y = load_all()

