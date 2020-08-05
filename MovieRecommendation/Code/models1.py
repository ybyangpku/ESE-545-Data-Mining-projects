"""
Created on April 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt


class Epsilon_Greedy_Partial:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the mean
        self.mu = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        epsilon = self.N/(1 + t) # 1 / (1 + t)
        p_unif = np.random.uniform()
        if epsilon > p_unif:
            idx = np.random.randint(self.N)
        else:
            idx = np.argmax(self.mu)
            
          
        reward_curr = self.X[idx, t]
        self.counts[idx] += 1
        self.mu[idx] = self.mu[idx] + (reward_curr - self.mu[idx]) / self.counts[idx]
        
        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        mean = self.mu.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx




class UCB_Partial:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the mean
        self.mu = np.zeros((self.N, 1))
        
        # store std
        self.std = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.ones((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        beta = 1./10
        self.std = np.sqrt(beta*np.log(t + 1)/self.counts)
        
        idx = np.argmax(self.mu + self.std)
            
        reward_curr = self.X[idx, t]
        self.counts[idx] += 1
        self.mu[idx] = self.mu[idx] + (reward_curr - self.mu[idx]) / self.counts[idx]
        
        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr
        
        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t + self.N], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t + self.N], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards
    

    def train(self):   
        
        loss = 0
        total_rewards = 0

        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
        
        random_idx = np.random.permutation(self.N)
        # initial push
        for it in range(self.N):
            self.mu[it] = self.X[random_idx[it], it]
            reward_curr = self.X[random_idx[it], it]
            
            loss = loss + 1 - reward_curr
            total_rewards = total_rewards + reward_curr

        best_reward = np.max(np.sum(self.X[:,:it], axis = 1))
        regret_curr = best_reward - total_rewards
        best_loss = np.min(np.sum(1 - self.X[:,:it], axis = 1))

            
        self.regrets.append(regret_curr)
        self.losses.append(loss)
        self.best_losses.append(best_loss)
        self.total_rewards.append(total_rewards)
            
        # UCB part    
        for it in range(self.T - self.N):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        mean = self.mu.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx




class Thompson_Sampling_Partial:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the number of successes
        self.S = np.zeros((self.N, 1))
        
        # store the number of failures
        self.F = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        gamma = 10
        theta = np.random.beta(self.S + 1, self.F + 1)
        
        idx = np.argmax(theta)
                    
        reward_curr = self.X[idx, t]
        if reward_curr == 1:
            self.S[idx] += 1 * gamma
        else:
            self.F[idx] += 1 * gamma
            
        self.counts[idx] += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
            
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    
    

    def get_best_arm(self, N_best): 
        theta = (self.S + 1)/ (self.S + self.F + 2) 
        mean = theta.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx





class EXP3:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
                        
        # number of movies
        self.N = X.shape[0]
        
        # candidate_index
        self.candidates = np.arange(self.N)
        
        # store data
        self.X = X
                
        # store the mean
        self.weights = np.ones((self.N, 1))

        # cumulative loss
        self.closs = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        delta = 100.
        eta = np.sqrt(1./ (t/delta+1)) # np.sqrt(np.log(self.N)/ (t+1)/self.N)
        
        Phi_t = np.sum(self.weights)
        
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = np.random.choice(self.candidates, 1, p=prob)[0]
#        print(prob[idx])
#        print("max prob", np.max(prob))
        
        indicator = np.zeros((self.N, 1))
        indicator[idx] = 1
        
        prob = self.weights / Phi_t
        reward_curr = self.X[idx, t]
        cost_for_this_round = (1 - self.X[:, t][:,None]) * indicator / prob
#        print("cost", np.sum(cost_for_this_round))
        
        self.closs += cost_for_this_round
#        print("min loss", np.min(self.closs))
        
        ##### this line maybe very useful
        temp_closs = self.closs - np.min(self.closs)

        self.weights = np.exp(-eta * temp_closs) + 1e-200

        self.counts[idx] += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = prob.argsort()[-N_best:][::-1]
        return idx



class Multiplicative_weight_update_Partial:
    # Initialize the class
    def __init__(self, X):  
      
        # number of top movies
        self.N_best = 10
        
        # number of days
        self.T = X.shape[1]
        
        # define eta
        self.eta = 1./np.sqrt(self.T)
        
        # number of movies
        self.N = X.shape[0]
        
        # candidate_index
        self.candidates = np.arange(self.N)
        
        # store data
        self.X = X
        
        # store the mean
        self.weights = np.ones((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # top indexs
        here = np.sum(self.X, axis = 1) / self.T
        self.top_index = here.argsort()[-self.N_best:][::-1]
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = np.random.choice(self.candidates, 1, p=prob)[0]
            
        indicator = np.zeros((self.N, 1))
        indicator[idx] = 1
        
        prob = self.weights / Phi_t        
        
        reward_curr = self.X[idx, t]
        cost_for_this_round = (1 - self.X[:, t][:,None]) * indicator / prob
        
        self.weights = self.weights * (1 - self.eta * cost_for_this_round)
        self.weights = (abs(self.weights) + self.weights) / 2. + 1e-300
        
        
        self.counts[idx] += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = prob.argsort()[-N_best:][::-1]
        return idx







class Epsilon_Greedy_Full:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the mean
        self.mu = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
        
    
    def update(self, X, t): 
        epsilon = 1/(1 + t)
        p_unif = np.random.uniform()
        if epsilon > p_unif:
            idx = np.random.randint(self.N)
        else:
            idx = np.argmax(self.mu)
        
        reward_curr = self.X[idx, t]
        rewards_for_this_round = self.X[:, t][:,None]
        self.counts += 1
        self.mu = self.mu + (rewards_for_this_round - self.mu) / self.counts

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        mean = self.mu.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx







class UCB_Full:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the mean
        self.mu = np.zeros((self.N, 1))
        
        # store std
        self.std = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
        
        
    def update(self, X, t): 
        
        self.std = np.sqrt(2*np.log(t + 1)/self.counts)
        
        idx = np.argmax(self.mu + self.std)
            
        reward_curr = self.X[idx, t]
        rewards_for_this_round = self.X[:, t][:,None]
        self.counts += 1
        self.mu = self.mu + (rewards_for_this_round - self.mu) / self.counts

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    
    

    def get_best_arm(self, N_best): 
        mean = self.mu.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx



class Thompson_Sampling_Full:
    # Initialize the class
    def __init__(self, X):  
      
        # number of days
        self.T = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # store the number of successes
        self.S = np.zeros((self.N, 1))
        
        # store the number of failures
        self.F = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        theta = np.random.beta(self.S + 1, self.F + 1)
        
        idx = np.argmax(theta)
            
        reward_curr = self.X[idx, t]
        rewards_for_this_round = self.X[:, t][:,None]

        indicator = (rewards_for_this_round == 1)
        
        self.S += indicator
        self.F += 1 - indicator
            
        self.counts += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    
    

    def get_best_arm(self, N_best): 
        theta = (self.S + 1)/ (self.S + self.F + 2) 
        mean = theta.flatten()
        idx = mean.argsort()[-N_best:][::-1]
        return idx




class EXP3_Full:
    # Initialize the class
    def __init__(self, X):  
      
        # number of top movies
        self.N_best = 10
    
        # number of days
        self.T = X.shape[1]
                        
        # number of movies
        self.N = X.shape[0]
        
        # candidate_index
        self.candidates = np.arange(self.N)
        
        # store data
        self.X = X
                
        # store the mean
        self.weights = np.ones((self.N, 1))

        # cumulative loss
        self.closs = np.zeros((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # top indexs
        here = np.sum(self.X, axis = 1) / self.T
        self.top_index = here.argsort()[-self.N_best:][::-1]
        
        # top probability 
        self.prob_record = np.zeros((self.N_best, self.T))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        eta = np.sqrt(1/(t/10+1)) # np.sqrt(np.log(self.N)/ (t+1)/self.N)
        
        Phi_t = np.sum(self.weights)
        
        prob = self.weights / Phi_t
        prob = prob.flatten()
        self.prob_record[:,t:t+1] = prob[self.top_index][:,None]
        
        idx = np.random.choice(self.candidates, 1, p=prob)[0]

        reward_curr = self.X[idx, t]
        cost_for_this_round = (1 - self.X[:, t][:,None])
        
        self.closs += cost_for_this_round
        
        temp_closs = self.closs - np.min(self.closs)

        self.weights = np.exp(-eta * temp_closs)

        self.counts += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = prob.argsort()[-N_best:][::-1]
        return idx




class Multiplicative_weight_update_Full:
    # Initialize the class
    def __init__(self, X):  
      
        # number of top movies
        self.N_best = 10
        
        # number of days
        self.T = X.shape[1]
        
        # define eta
        self.eta = 1./np.sqrt(self.T) # 3. / np.sqrt(self.T)
        
        # number of movies
        self.N = X.shape[0]
        
        # candidate_index
        self.candidates = np.arange(self.N)
        
        # store data
        self.X = X
        
        # store the mean
        self.weights = np.ones((self.N, 1))
        
        # store the number of push
        self.counts = np.zeros((self.N, 1))
        
        # top indexs
        here = np.sum(self.X, axis = 1) / self.T
        self.top_index = here.argsort()[-self.N_best:][::-1]
        
        # top probability 
        self.prob_record = np.zeros((self.N_best, self.T))
        
        # Loss 
        self.regrets = []
        self.losses = []
        self.best_losses = [] 
        self.total_rewards = []
    
    def update(self, X, t): 
        
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        self.prob_record[:,t:t+1] = prob[self.top_index][:,None]
        
        idx = np.random.choice(self.candidates, 1, p=prob)[0]
            
        reward_curr = self.X[idx, t]
        rewards_for_this_round = self.X[:, t][:,None]
        cost_for_this_round = 1 - rewards_for_this_round
        self.weights = self.weights * (1 - self.eta * cost_for_this_round)
        self.weights = (abs(self.weights) + self.weights) / 2. + 1e-300
        
        self.counts += 1

        loss = self.losses[-1]  
        loss_curr = loss + 1 - reward_curr
        total_rewards = self.total_rewards[-1]
        total_rewards = total_rewards + reward_curr

        if t % 50 == 0:
            best_reward = np.max(np.sum(self.X[:,:t], axis = 1))
            regret_curr = best_reward - total_rewards

            best_loss = np.min(np.sum(1 - self.X[:,:t], axis = 1))

            return regret_curr, loss_curr, best_loss, total_rewards
        else:
            return 0, loss_curr, 0, total_rewards

    
    def train(self):   
        
        self.regrets.append(0)
        self.losses.append(0)
        self.best_losses.append(0)
        self.total_rewards.append(0)
                
        for it in range(self.T):
                
            regret_curr, loss_curr, best_loss, total_rewards = self.update(self.X, it)
            
            self.losses.append(loss_curr)
            self.total_rewards.append(total_rewards)
            
            if it % 50 == 0:
                self.regrets.append(regret_curr)
                self.best_losses.append(best_loss)


            # Print
            if it % 1000 == 0:
                print('It: %d, regret: %.3e, loss: %.3e, best_loss: %.3e, total_rewards: %.3e' % 
                      (it, regret_curr, loss_curr, best_loss, total_rewards))
    

    def get_best_arm(self, N_best): 
        Phi_t = np.sum(self.weights)
        prob = self.weights / Phi_t
        prob = prob.flatten()
        
        idx = prob.argsort()[-N_best:][::-1]
        return idx




