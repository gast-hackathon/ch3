

import numpy as np
import utils
import models
import data

def _search(model, country_code, with_index=False):
	
	dataset = data.Dataset(country_code, with_index)

	avg, avg_y = dataset.country_average()

	if with_index:
		base = 9
	else:
		base = 0

	avg_applications = avg[0, base]
	avg_rejections = avg[0, base+1]
	avg_fraud = avg_y

	print("country:", country_code)
	print("\taverage applications", avg_applications)
	print("\taverage rejections", avg_rejections)
	print("\taverage fraud", avg_fraud)

	rejections = 0

	while rejections < avg_applications:

		avg[0, base+1] = rejections

		prediction = model.predict(avg)

		print(rejections, prediction)

		rejections += 100

def exhaustive_search(model, full_dataset, country_code):

	data.set_full_dataset(full_dataset)

	own_model = data.Dataset(country_code).to_model()
	own_model.train()

	#_search(model, country_code, True)
	_search(own_model, country_code)

	





