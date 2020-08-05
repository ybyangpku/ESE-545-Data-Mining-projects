"""
Created on Feb 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from scipy.spatial import distance

from models2 import Kmeans_online, Kmeanspp_online

#np.random.seed(1234)
    
if __name__ == "__main__":

	# Question 1
	Data = pd.read_csv("Movies.csv", header=None) #, sep=',')
	Data = np.asarray(Data)
	Data = np.hstack((Data[:,3][:,None], Data[:,6:]))
	Data = Data[1:,:]
	Data = np.float32(Data)


	print(np.sum(Data))
	print(Data.shape)


	N_batch = 10000
	epochs = 1000

	### Question 2

	num_class = [5, 10, 20, 50, 100, 200, 400, 500]
	N_total = len(num_class)
	final_minimum = np.zeros((N_total, 1))
	final_mean_value = np.zeros((N_total, 1))
	final_maximum = np.zeros((N_total, 1))

	for k in range(N_total):
		model = Kmeans_online(Data, num_class[k])
		model.train(500, 10000)

		minimum, mean_value, maximum = model.get_final_error()
		final_minimum[k] = minimum
		final_mean_value[k] = mean_value
		final_maximum[k] = maximum

		print(minimum, mean_value, maximum)

	plt.figure(1, figsize=(6,4))
	plt.plot(num_class, final_minimum, label = "minimum")
	plt.plot(num_class, final_mean_value, label = "mean_value")
	plt.plot(num_class, final_maximum, label = "maximum")
	plt.legend()
	plt.savefig('./kmeans_error.png', dpi = 100)




	### Question 3

	num_class = [5, 10, 20, 50, 100, 200, 400, 500]
	N_total = len(num_class)
	final_minimum_pp = np.zeros((N_total, 1))
	final_mean_value_pp = np.zeros((N_total, 1))
	final_maximum_pp = np.zeros((N_total, 1))


	for k in range(N_total):
		model = Kmeanspp_online(Data, num_class[k])
		model.train(500, 10000)

		minimum, mean_value, maximum = model.get_final_error()
		final_minimum_pp[k] = minimum
		final_mean_value_pp[k] = mean_value
		final_maximum_pp[k] = maximum

		print(minimum, mean_value, maximum)


	plt.figure(2, figsize=(6,4))
	plt.plot(num_class, final_minimum_pp, label = "minimum")
	plt.plot(num_class, final_mean_value_pp, label = "mean_value")
	plt.plot(num_class, final_maximum_pp, label = "maximum")
	plt.legend()
	plt.savefig('./kmeanspp_error.png', dpi = 100)





