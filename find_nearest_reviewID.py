"""
Created on Feb 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


if __name__ == "__main__":
    
    #### test the code in line 230  ####
	r = 2
	b = 300
	M = 600
	k_num = 5
	P = np.load("P.npy")
	R = np.load("R.npy")

	pi = np.load("pi.npy")
	pi_r = np.load("pi_r.npy")

	dataset = pd.read_pickle('dataset.pkl')

    #######################
	# functions for removeing the punctuations and stop words
	#######################
	# All possible punctuations
	punctuations = '''!()-=+|[]{};:'"\,<>./?@#$%^&*_~'''

	# All possible stop words
	stop_words_list = []
	with open('stop_words_list.txt') as data_file:
	    data=data_file.read()
	    for i in data.split('\n'):
	        stop_words_list.append(i)
	        
	# Parse the input sentence by removing all punctuations and stop words (one sentence each time)
	def parse_sentence(sentence, punctuations, stop_words_list):
	    # First remove punctuations by looping each char
	    no_punct = ""
	    for i in sentence:
	        if i not in punctuations:
	            no_punct = no_punct + i
	            
	    # Second remove stop words by looping each words
	    final_sentence = ""
	    for word in no_punct.split():
	        if word not in stop_words_list:
	            # See if we are at the start of the sentence 
	            if final_sentence == "":
	                final_sentence = word
	            else:
	                final_sentence = final_sentence + " " + word
	            
	    return final_sentence


    #######################
	# load k_shingles_dict
	#######################
	reader = csv.reader(open('shingle_dict.csv'))

	k_shingles_dict = {}
	for row in reader:
	    key = row[0]
	    if key in k_shingles_dict:
	        # implement your duplicate row handling here
	        pass
	    here = int(row[1:][0])
	    k_shingles_dict[key] = here


	n_row = len(k_shingles_dict)


    #######################
	# load list_buckets
	#######################
	print("load list_buckets")
	num_buckets = b

	import json
	with open("list_buckets.json", "r") as f:
		get_list_buckets = json.load(f)


	list_buckets = []

	for i in range(num_buckets):
	    temp_buckets = {}
	    for key in get_list_buckets[i]:
	        temp_buckets[float(key)] = get_list_buckets[i][key]
	    list_buckets.append(temp_buckets)
	    
    


	def h_pi(data, pi, R, M):
	    data = np.asarray(data)[None,:]
	    aa = pi[:,0:1]
	    bb = pi[:,1:2]
	    hash_vector = np.min((aa * data + bb) % R  ,axis=1)[:,None]
	    return hash_vector


	# generate vector for sentence index = i
	def generate_binary_vector(i):
	    # Load one sentence 
	    sentence = dataset["Text"][i]
	    
	    Binary_data_vector = np.zeros((n_row, 1))
	    # Compute the length of the sentence
	    length = len(sentence)
	    
	    # If the sentence length is less than 50, do padding with 50 space
	    if length < k_num:
	        sentence = sentence + " " * k_num
	        length += k_num
	    
	    # Loop the sentence
	    for j in range(length - k_num + 1):
	        temp = sentence[j:j+k_num]
	        if temp in k_shingles_dict:
	            Binary_data_vector[k_shingles_dict[temp]] = 1

	    return Binary_data_vector


	# generate vector for sentence index = i
	def convert_binary_vector(sentence):
	    # Load one sentence 
	    
	    Binary_data_vector = np.zeros((n_row, 1))
	    # Compute the length of the sentence
	    length = len(sentence)
	    
	    # If the sentence length is less than 50, do padding with 50 space
	    if length < k_num:
	        sentence = sentence + " " * k_num
	        length += k_num
	    
	    # Loop the sentence
	    for j in range(length - k_num + 1):
	        temp = sentence[j:j+k_num]
	        if temp in k_shingles_dict:
	            Binary_data_vector[k_shingles_dict[temp]] = 1

	    return Binary_data_vector


	def find_the_nearest(list_buckets, queried_review, b, r):
	    # parsing the review by removing the punctuations and stop words
	    sentence = parse_sentence(queried_review.lower(), punctuations, stop_words_list)
	    
	    # compute the indexs of the none zeros elements and put it into a list
	    temp_list = []
	    seen = set()
	    
	    length = len(sentence)
	    
	    if length < k_num:
	        sentence = sentence + " " * k_num
	        length += k_num
	    
	    for j in range(length - k_num + 1):
	        temp = sentence[j:j+k_num]
	        if temp in k_shingles_dict and temp not in seen:
	            seen.add(temp)
	            temp_list.append(k_shingles_dict[temp])
	    
	    # compute the signature vector 
	    signature_vector = h_pi(temp_list, pi, R, M)
	    
	    # compute the bucket value for each band and search in the bucket_list we got from the previous question
	    nearest_index = []
	    
	    a_v = pi_r[:, 0:1]
	    b_v = pi_r[:, 1:2]
	    for i in range(b):
	        data = signature_vector[i*r:(i+1)*r,0:1]
	        temp = np.sum((data * a_v + b_v) % P)
	        bucket = list_buckets[i]
	        if temp in bucket:
	            num = len(bucket[temp])
	            for j in range(num):
	                nearest_index.append(bucket[temp][j])
	        
	    # find the nearest index and concatenate them together and return 
	    return_ID = []
	    num_nearest = len(nearest_index)
	    for i in range(num_nearest):
	        if dataset["ID"][nearest_index[i]] not in return_ID:
	            return_ID.append(dataset["ID"][nearest_index[i]])
	    
	    length = len(nearest_index)
	    if length > 0:
	        sentence_1 = convert_binary_vector(sentence)
	        Jaccard_distance_min = 1.
	        id_min = nearest_index[0]

	        for i in range(length):
	            idx2 = nearest_index[i]
	            sentence_2 = generate_binary_vector(idx2)

	            count_and = np.sum(sentence_1 * sentence_2)
	            count_or = np.sum(sentence_1) + np.sum(sentence_2) - count_and

	            Jaccard_distance = 1 - count_and / count_or

	            if Jaccard_distance < Jaccard_distance_min:
	                Jaccard_distance_min = Jaccard_distance
	                id_min = idx2
	    else:
	        id_min = -1
	        
	    if id_min == -1:
	        final_nearest_ID = "Cannot find any similar review having Jaccard distance less than 0.8 with the given one"
	    else:
	        final_nearest_ID = dataset["ID"][id_min]
	    
	    return return_ID, id_min, final_nearest_ID


	############# test here ############

	print("Show some test cases for finding the nearest neighbor")

	print("\n")

	# test some different cases
	queried_review = "good quality dental chew less give one day freshens dogs breath quality edible ingredients"

	return_ID, idx, id_min = find_the_nearest(list_buckets, queried_review, b, r)
	print("Inpute queried_review is:", queried_review)
	print("The closest reviewID of the given queried_review is:", id_min)
	if idx != -1:
		print("It is:", dataset["Text"][idx])


	print("\n")

	queried_review = "dog loved birthday loves chance havent gotten see durability yet since days floating far hasnt chewed"

	return_ID, idx, id_min = find_the_nearest(list_buckets, queried_review, b, r)
	print("Inpute queried_review is:", queried_review)
	print("The closest reviewID of the given queried_review is:", id_min)
	if idx != -1:
		print("It is:", dataset["Text"][idx])

	print("\n")

	queried_review = "I have a nice good cat walk around me."

	return_ID, idx, id_min = find_the_nearest(list_buckets, queried_review, b, r)
	print("Inpute queried_review is:", queried_review)
	print("The closest reviewID of the given queried_review is:", id_min)
	if idx != -1:
		print("It is:", dataset["Text"][idx])


	print("\n")

	queried_review = "I like to have enough time walking my dog run for long time"

	return_ID, idx, id_min = find_the_nearest(list_buckets, queried_review, b, r)
	print("Inpute queried_review is:", queried_review)
	print("The closest reviewID of the given queried_review is:", id_min)
	if idx != -1:
		print("It is:", dataset["Text"][idx])



		

