"""
Created on Feb 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt


from models1 import Epsilon_Greedy_Partial, UCB_Partial, Thompson_Sampling_Partial, EXP3, Multiplicative_weight_update_Partial, Epsilon_Greedy_Full, UCB_Full, Thompson_Sampling_Full, EXP3_Full, Multiplicative_weight_update_Full

#np.random.seed(1234)
    
if __name__ == "__main__":

	###### Question 1 ######
	Data = pd.read_csv("recommendationMovie.csv", header=None) #, sep=',')
	Data = np.asarray(Data)
	print(Data.shape)
	print(np.sum(Data))





	###### Question 2 ######

	# epsilon_greedy_partial
	model_epsilon_greedy_partial = Epsilon_Greedy_Partial(Data)
	model_epsilon_greedy_partial.train()

	idx = model_epsilon_greedy_partial.get_best_arm(10)

	regrets = model_epsilon_greedy_partial.regrets
	losses = model_epsilon_greedy_partial.losses[::50]
	best_losses = model_epsilon_greedy_partial.best_losses

	iterations1 = 50 * np.arange(len(losses))
	iterations2 = 50 * np.arange(len(best_losses))


	print(model_epsilon_greedy_partial.mu)
	print(model_epsilon_greedy_partial.counts)
	print("best movies", idx)


	plt.figure(11, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_epsilon_greedy_partial.png', dpi = 100)


	plt.figure(12, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_epsilon_greedy_partial.png', dpi = 100)


	# UCB_partial
	model_UCB_partial = UCB_Partial(Data)
	model_UCB_partial.train()

	idx = model_UCB_partial.get_best_arm(10)
	regrets = model_UCB_partial.regrets
	losses = model_UCB_partial.losses[::50]
	best_losses = model_UCB_partial.best_losses

	iterations1 = 50 * np.arange(len(losses))
	iterations2 = 50 * np.arange(len(best_losses))


	print(model_UCB_partial.mu)
	print(model_UCB_partial.counts)
	print("best movie", idx)


	plt.figure(21, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_UCB_partial.png', dpi = 200)

	plt.figure(22, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_UCB_partial.png', dpi = 100)



	# Thompson_Sampling_Partial
	model_Thompson_Sampling_Partial = Thompson_Sampling_Partial(Data)
	model_Thompson_Sampling_Partial.train()

	idx = model_Thompson_Sampling_Partial.get_best_arm(10)
	regrets = model_Thompson_Sampling_Partial.regrets
	losses = model_Thompson_Sampling_Partial.losses[::50]
	best_losses = model_Thompson_Sampling_Partial.best_losses


	iterations1 = 50 * np.arange(len(losses))
	iterations2 = 50 * np.arange(len(best_losses))


	print(model_Thompson_Sampling_Partial.counts)
	print("best movie", idx)


	plt.figure(31, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_Thompson_Sampling_Partial.png', dpi = 200)

	plt.figure(32, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_Thompson_Sampling_Partial.png', dpi = 100)



	# EXP3 
	model_EXP3 = EXP3(Data)
	model_EXP3.train()

	idx = model_EXP3.get_best_arm(10)
	regrets = model_EXP3.regrets
	losses = model_EXP3.losses[::50]
	best_losses = model_EXP3.best_losses


	print(model_EXP3.counts)
	print("best movie", idx)


	plt.figure(41, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_EXP3.png', dpi = 200)

	plt.figure(42, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_EXP3.png', dpi = 100)





	# Multiplicative weight update Partial 
	model_Multiplicative_weight_update_Partial = Multiplicative_weight_update_Partial(Data)
	model_Multiplicative_weight_update_Partial.train()

	idx = model_Multiplicative_weight_update_Partial.get_best_arm(10)
	regrets = model_Multiplicative_weight_update_Partial.regrets
	losses = model_Multiplicative_weight_update_Partial.losses[::50]
	best_losses = model_Multiplicative_weight_update_Partial.best_losses


	print(model_Multiplicative_weight_update_Partial.counts)
	print("best movie", idx)


	plt.figure(51, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_Multiplicative_weight_update_Partial.png', dpi = 200)

	plt.figure(52, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_Multiplicative_weight_update_Partial.png', dpi = 100)
	plt.close('all')











	###### Question 3 ######

	# epsilon_greedy_full 
	model_epsilon_greedy_full = Epsilon_Greedy_Full(Data)
	model_epsilon_greedy_full.train()

	idx = model_epsilon_greedy_full.get_best_arm(10)
	regrets = model_epsilon_greedy_full.regrets
	losses = model_epsilon_greedy_full.losses[::50]
	best_losses = model_epsilon_greedy_full.best_losses

	print(model_epsilon_greedy_full.mu)
	print(model_epsilon_greedy_full.counts)
	print("best movie", idx)


	plt.figure(101, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_epsilon_greedy_full.png', dpi = 100)

	plt.figure(102, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_epsilon_greedy_full.png', dpi = 100)




	# UCB_full
	model_UCB_full = UCB_Full(Data)
	model_UCB_full.train()

	idx = model_UCB_full.get_best_arm(10)
	regrets = model_UCB_full.regrets
	losses = model_UCB_full.losses[::50]
	best_losses = model_UCB_full.best_losses


	print(model_UCB_full.mu)
	print(model_UCB_full.counts)
	print("best movie", idx)


	plt.figure(201, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_UCB_full.png', dpi = 200)

	plt.figure(202, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_UCB_full.png', dpi = 100)






	# Thompson_Sampling_full 
	model_Thompson_Sampling_full = Thompson_Sampling_Full(Data)
	model_Thompson_Sampling_full.train()

	idx = model_Thompson_Sampling_full.get_best_arm(10)
	regrets = model_Thompson_Sampling_full.regrets
	losses = model_Thompson_Sampling_full.losses[::50]
	best_losses = model_Thompson_Sampling_full.best_losses


	print(model_Thompson_Sampling_full.counts)
	print("best movie", idx)


	plt.figure(301, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_Thompson_Sampling_full.png', dpi = 200)

	plt.figure(302, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_Thompson_Sampling_full.png', dpi = 100)



	# EXP3 Full
	model_EXP3_Full = EXP3_Full(Data)
	model_EXP3_Full.train()

	idx = model_EXP3_Full.get_best_arm(10)
	regrets = model_EXP3_Full.regrets
	losses = model_EXP3_Full.losses[::50]
	best_losses = model_EXP3_Full.best_losses


	print(model_EXP3_Full.counts)
	print("best movie", idx)


	plt.figure(401, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_EXP3_Full.png', dpi = 100)

	plt.figure(402, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_EXP3_Full.png', dpi = 100)


	gap = 10
	top_probs = model_EXP3_Full.prob_record
	top_probs = np.transpose(top_probs[:,::gap])
	iterations = np.arange(Data.shape[1])
	iterations = iterations[::gap]

	plt.figure(403, figsize=(8,5.5))
	for k in range(10):
	    plt.plot(iterations, top_probs[:,k], label = "top" + str(k+1) + "movie")
	plt.legend()
	plt.savefig('./top_10_movies_EXP3.png', dpi = 100)



	# Multiplicative_weight_update_Full
	model_Multiplicative_weight_update_Full = Multiplicative_weight_update_Full(Data)
	model_Multiplicative_weight_update_Full.train()

	idx = model_Multiplicative_weight_update_Full.get_best_arm(10)
	regrets = model_Multiplicative_weight_update_Full.regrets
	losses = model_Multiplicative_weight_update_Full.losses[::50]
	best_losses = model_Multiplicative_weight_update_Full.best_losses


	print(model_Multiplicative_weight_update_Full.counts)
	print("best movie", idx)


	plt.figure(501, figsize=(6,4))
	plt.plot(iterations2, regrets, label = "regrets")
	plt.legend()
	plt.savefig('./regrets_model_Multiplicative_weight_update_Full.png', dpi = 200)

	plt.figure(502, figsize=(6,4))
	plt.plot(iterations1, losses, label = "losses")
	plt.plot(iterations2, best_losses, label = "optimal_losses")
	plt.legend()
	plt.savefig('./losses_model_Multiplicative_weight_update_Full.png', dpi = 100)


	gap = 10
	top_probs = model_Multiplicative_weight_update_Full.prob_record
	top_probs = np.transpose(top_probs[:,::gap])
	iterations = np.arange(Data.shape[1])
	iterations = iterations[::gap]

	plt.figure(503, figsize=(8,5.5))
	for k in range(10):
	    plt.plot(iterations, top_probs[:,k], label = "top" + str(k+1) + "movie")
	plt.legend()
	plt.savefig('./top_10_movies_Multiplicative_weight_update.png', dpi = 100)
	plt.close('all')




