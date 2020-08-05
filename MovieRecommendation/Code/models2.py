"""
Created on April 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Kmeans_online:
    # Initialize the class
    def __init__(self, X, num_class):  
      
        # number of days
        self.dim = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # number of clusters
        self.num_class = num_class
        
        # Initial centers
        self.centers = self.initial_centers()
        
        # Loss 
        self.errors = []
        
        
    def initial_centers(self):
        ub = np.max(self.X, axis = 0)
        lb = np.min(self.X, axis = 0)
        centers = np.zeros((self.num_class, self.dim))
        for k in range(self.num_class):
            for p in range(self.dim):
                centers[k, p] = np.random.uniform() * (ub[p] - lb[p]) + lb[p]
        
        return centers

        
    def update(self, X, it): 
        
        eta = 1./(it+1)
        error_curr = 0
        N_batch = X.shape[0]
        temp_dist = np.zeros((self.num_class, 1))
        centers_grad = np.zeros_like(self.centers)
        
        distance_matrix = distance.cdist(X, self.centers, 'euclidean')
        idx = np.argmin(distance_matrix, axis = 1)
        
        for k in range(N_batch):
            centers_grad[idx[k]:idx[k]+1,:] += 2 * (X[k:k+1,:] - self.centers[idx[k]:idx[k]+1,:])
            
        self.centers += eta * centers_grad / N_batch * self.num_class

        distance_matrix = distance.cdist(X, self.centers, 'euclidean')
        error_curr = np.sum(np.min(distance_matrix, axis = 1))
        return error_curr
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        return X_batch
    

    def train(self, T, N_batch):   
        
                
        for it in range(T):
            X_batch = self.fetch_minibatch(self.X, N_batch)
            error = self.update(X_batch, it)
            
            self.errors.append(error)

            # Print
            if it % 10 == 0:
                print('It: %d, error: %.3e' % 
                      (it, error))
    
    
    def get_centers(self): 
        return self.centers
    
    def get_final_error(self):
        distance_matrix = distance.cdist(self.X, self.centers, 'euclidean')
        distances = np.min(distance_matrix, axis = 1)
        error = np.sum(distances)
        minimum = np.min(distances)
        maximum = np.max(distances)
        mean_value = np.mean(distances)
        return minimum, mean_value, maximum
    

    




class Kmeanspp_online:
    # Initialize the class
    def __init__(self, X, num_class):  
      
        # number of days
        self.dim = X.shape[1]
        
        # number of movies
        self.N = X.shape[0]
        
        # store data
        self.X = X
        
        # number of clusters
        self.num_class = num_class
        
        # Initial centers
        self.centers = self.initial_centers()
        
        # Loss 
        self.errors = []

        
    def initial_centers(self):
        idx = np.random.randint(self.N)
        prob = np.zeros((self.N, 1))
        temp_dist = np.zeros((self.num_class, 1))
        candidates = np.arange(10000)  # np.arange(self.N)
        
        centers = self.X[idx:idx+1,:]
        
        X = self.fetch_minibatch(self.X, 10000)
        
        for k in range(1, self.num_class):
            distance_matrix = distance.cdist(X, centers, 'euclidean')**2
            prob = np.min(distance_matrix, axis = 1)
            prob = prob / np.sum(prob)
            idx = np.random.choice(candidates, 1, p=prob)[0]
            centers_temp = X[idx:idx+1,:]
            centers = np.vstack((centers, centers_temp))
            
        return centers

        
    def update(self, X, it): 
        
        eta = 1/(it+1)
        error_curr = 0
        N_batch = X.shape[0]
        temp_dist = np.zeros((self.num_class, 1))
        centers_grad = np.zeros_like(self.centers)
        
        distance_matrix = distance.cdist(X, self.centers, 'euclidean')
        idx = np.argmin(distance_matrix, axis = 1)
        
        for k in range(N_batch):
            centers_grad[idx[k]:idx[k]+1,:] += 2 * (X[k:k+1,:] - self.centers[idx[k]:idx[k]+1,:])
            
        self.centers += eta * centers_grad / N_batch

        distance_matrix = distance.cdist(X, self.centers, 'euclidean')
        error_curr = np.sum(np.min(distance_matrix, axis = 1))
        return error_curr
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        return X_batch
    

    def train(self, T, N_batch):   
                
        for it in range(T):
            X_batch = self.fetch_minibatch(self.X, N_batch)
            error = self.update(X_batch, it)
            
            self.errors.append(error)

            # Print
            if it % 10 == 0:
                print('It: %d, error: %.3e' % 
                      (it, error))
    
    
    def get_centers(self): 
        return self.centers
    
    def get_final_error(self):
        distance_matrix = distance.cdist(self.X, self.centers, 'euclidean')
        distances = np.min(distance_matrix, axis = 1)
        error = np.sum(distances)
        minimum = np.min(distances)
        maximum = np.max(distances)
        mean_value = np.mean(distances)
        return minimum, mean_value, maximum
    
