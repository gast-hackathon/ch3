

import xlrd
from xlrd.sheet import Sheet


def col_x(col):

	return [
		col[2] + col[4],  # 0. Applications
		col[22],          # 1. Rejections
		col[8],           # 2. Never payer
		col[9],           # 3. Churn rate
		col[10],          # 4. Forced disconnection
		col[11],          # 5. Fraud count - subscriber
		col[12],          # 6. Fraud amount - receivables
		col[13],          # 7. Fraud amount - outside receivables
		col[14],          # 8. Average amount per fraud case
		col[15],          # 9. Subscribers blocked - fraud
		col[16],          # 10. Subscribers blocked - all
		col[17],          # 11. Balance blocked - credit monitoring
	]


def load_country(country_code):

	wb = xlrd.open_workbook("DTSE_Hackathon_C3_Credit_Forecast_data.xlsx")

	sheet = wb.sheet_by_name(country_code)

	for col in range(sheet.ncols - 1):
		col_values = sheet.col_values(col + 1)
		x = col_x(col_values)
		y = col_values[]


if __name__ == "__main__":
	load_country("AT")

