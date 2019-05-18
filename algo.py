

import numpy as np
import utils
import models
import data

import matplotlib.pyplot as plt

fig = 0

def _search(model, country_code, with_index=False):
	
	dataset = data.Dataset(country_code, with_index)

	avg, avg_y = dataset.country_average()

	if with_index:
		base = 9
	else:
		base = 0

	avg_applications = avg[0, base]
	avg_rejections = avg[0, base+1]
	avg_accepted = avg_applications * (1 - avg_rejections)
	avg_fraud = avg_y * avg_accepted

	plt_x = []
	plt_fraud = []
	plt_payers = []
	plt_avg_fraud = []

	rej_rate = 0

	while rej_rate < 0.4:

		avg[0, base+1] = rej_rate

		fraud_rate_prediction = model.predict(avg)

		accepted_applications = avg_applications*(1-rej_rate)

		plt_x.append(rej_rate)
		plt_fraud.append(accepted_applications * fraud_rate_prediction)
		plt_payers.append(accepted_applications * (1-fraud_rate_prediction))
		plt_avg_fraud.append(avg_fraud)

		rej_rate += 0.01
	
	plt.xlabel('Rejections')
	plt.plot(plt_x, plt_fraud, 'r', label='Fraudulent users')
	plt.plot(plt_x, plt_avg_fraud, 'b', label='Average fraudulent users')
	plt.axvline(avg_rejections, label='Average rejection rate')
	plt.ylim(bottom=0)
	#plt.plot(plt_x, plt_payers, 'g', label='Paying users')	
	plt.legend() 
	

def exhaustive_search(model, full_dataset, country_code):

	global fig
	plt.figure(fig)

	data.set_full_dataset(full_dataset)

	own_model = data.Dataset(country_code).to_model()
	own_model.set_pca(4)
	own_model.set_epochs(300)
	own_model.train()

	plt.subplot(3, 1, 1)
	plt.title(country_code + ' - all data')
	_search(model, country_code, True)

	plt.subplot(3, 1, 2)
	own_model.plot_history(False)

	plt.subplot(3, 1, 3)
	plt.title(country_code + ' - own data')
	_search(own_model, country_code)

	fig += 1

def plot():
	plt.show()





