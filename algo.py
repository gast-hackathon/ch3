

import numpy as np
import utils
import models
import data

import matplotlib.pyplot as plt

fig = 0

def _search(model, country_code, plot_revenue=False, with_index=False):
	
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
	avg_payers = avg_accepted - avg_fraud

	avg_payer_revenue = data.country_average_payer_revenue(country_code)
	avg_fraud_revenue = avg_fraud * avg_payer_revenue
	avg_revenue = avg_payers * avg_payer_revenue

	plt_x = []
	plt_fraud = []
	plt_revenue = []

	plt_avg_fraud = []
	plt_avg_revenue = []

	rej_rate = 0

	revenue_at_avg = 0
	pred_fraud_at_avg = 0

	while rej_rate < 0.4:

		avg[0, base+1] = rej_rate

		fraud_rate_prediction = model.predict(avg)

		accepted_applications = avg_applications*(1-rej_rate)
		fraud_clients = accepted_applications * fraud_rate_prediction
		paying_users = accepted_applications * (1-fraud_rate_prediction)

		total_revenue = paying_users * avg_payer_revenue
		total_loss = fraud_clients * avg_payer_revenue

		plt_fraud.append(fraud_clients)
		plt_revenue.append(total_revenue - total_loss)

		plt_x.append(rej_rate)
		plt_avg_fraud.append(avg_fraud_revenue)
		plt_avg_revenue.append(avg_revenue)

		if rej_rate > avg_rejections and not pred_fraud_at_avg:
			revenue_at_avg = total_revenue
			pred_fraud_at_avg = fraud_rate_prediction * accepted_applications * avg_payer_revenue

		rej_rate += 0.01
	
	worst_fraud = plt_fraud[0] * avg_payer_revenue

	if plot_revenue:
		plt.plot(plt_x, plt_revenue, 'g', label='Revenue')	
		plt.plot(plt_x, plt_avg_revenue, label='Average revenue')
	else:
		plt.plot(plt_x, plt_fraud, 'r', label='Fraud customers')
		
	plt.xlabel('Rejection')
	plt.axvline(avg_rejections, label='Average rejection rate')
	plt.ylim(bottom=0)
	plt.legend() 
	
	if plot_revenue:
		country_gain = np.max(np.array(plt_revenue)) - revenue_at_avg
		country_gain_per_decrease = country_gain / (avg_rejections * 100)
		perc_gain = round(100 * country_gain / avg_revenue, 2)
		perc_gain_per_decrease = round(perc_gain / (avg_rejections * 100), 3)

		fraud_increase = worst_fraud - pred_fraud_at_avg
		fraud_increase_per_decrease = round(fraud_increase / (avg_rejections * 100), 3)
		perc_fraud_increase = round(100 * fraud_increase / pred_fraud_at_avg, 2)
		perc_fraud_increase_per_decrease = round(perc_fraud_increase / (avg_rejections * 100), 3)

		print('country', country_code)
		print('\trejection decrease', (avg_rejections * 100), '%')
		print('\tmax gain', country_gain)
		print('\tgain per 1%', country_gain_per_decrease)
		print('\tperc gain', perc_gain)
		print('\tperc gain per 1%', perc_gain_per_decrease)
		print('\tfraud increase', fraud_increase)
		print('\tfraud inrease per 1%', fraud_increase_per_decrease)
		print('\tperc fraud increase', perc_fraud_increase)
		print('\tperc fraud inrease per 1%', perc_fraud_increase_per_decrease)

def exhaustive_search(model, full_dataset, country_code):

	global fig
	plt.figure(fig)

	data.set_full_dataset(full_dataset)

	own_model = data.Dataset(country_code).to_model()
	own_model.set_pca(4)
	own_model.set_epochs(300)
	own_model.train()

	plt.subplot(3, 2, 1)
	plt.title(country_code + ' - all data')
	_search(model, country_code, plot_revenue=True, with_index=True)
	plt.subplot(3, 2, 2)
	plt.title(country_code + ' - all data')
	_search(model, country_code, with_index=True)

	plt.subplot(3, 2, 3)
	own_model.plot_history(False)

	plt.subplot(3, 2, 5)
	plt.title(country_code + ' - own data')
	_search(own_model, country_code, plot_revenue=True)
	plt.subplot(3, 2, 6)
	plt.title(country_code + ' - own data')
	_search(own_model, country_code)

	fig += 1

def plot():
	plt.show()





