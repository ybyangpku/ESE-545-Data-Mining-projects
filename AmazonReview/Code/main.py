"""
Created on Feb 2020

@author: Yibo Yang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#np.random.seed(1234)
    
if __name__ == "__main__":
    
	# Question 1
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
	# load the data set
	#######################
	with open('amazonReviews.json') as data_file:
	    data=data_file.read()
	    jdata = pd.read_json(data, lines = True)
	    
	    
	num_sentences = len(jdata)
	# List to store all the needed and parsed information
	here = []

	print("Parsing the data and monitor the process")
	count = 0
	for i in range(num_sentences):
	    # Get the ID of the review and lower the letters
	    ID = jdata['reviewerID'][i]

	    # Get the parsed review text 
	    Sentence = parse_sentence(jdata['reviewText'][i].lower(), punctuations, stop_words_list)

	    # Append the ID and review sentence together as one data point (remove the short ones)
	    if len(Sentence) > 2:
	        here.append((ID, Sentence))

	    # Count sentences parsed
	    count += 1
	    if count % 1000 == 0:
	        print(count)
	        
	num_sentences = len(here)



	#######################
	# save the useful informations in the dataset
	#######################
	dataset=pd.DataFrame(here,columns=['ID','Text'])

	num_sentences = dataset.shape[0]
	#dataset
	dataset.to_pickle("dataset.pkl")

	#######################
	# choose the shingle length and build the dictionary mapping shingle to index
	#######################
	k_num = 5

	k_shingles_dict = {}
	number_of_seen_shingles = 0

	print("Building the dictionary of shingle and monitor the process")
	# Loop through each sentence
	for i in range(num_sentences):
	    # Print the process
	    if i % 1000 == 0:
	        print(i)

	    # Load one sentence 
	    sentence = dataset["Text"][i]
	    
	    # Compute the length of the sentence
	    length = len(sentence)
	    
	    # If the sentence length is less than 50, do padding with 50 space
	    if length < k_num:
	        sentence = sentence + " " * k_num
	        length += k_num
	    
	    # Loop the sentence
	    for j in range(length - k_num + 1):
	        temp = sentence[j:j+k_num]
	        if temp not in k_shingles_dict:
	            k_shingles_dict[temp] = number_of_seen_shingles
	            number_of_seen_shingles += 1



	# Question 2
	#######################
	# Constructing the binary matrix (here I commented the code because creating this huge matrix will blow up some computer's memory
	# but this code runs well on my computer. As we do not use this part later on, I do not include it here. If you want to see the results,
	# you can uncomment this part and run)
	#######################
	n_row = len(k_shingles_dict)
	n_col = num_sentences

	print(n_row, n_col)

	# Build the Binary data matrix

	#Binary_data_matrix = np.zeros((n_row, n_col))

	# Loop through each sentence
	#for i in range(num_sentences):
	    # Print the process
	#    if i % 1000 == 0:
	#        print(i)

	    # Load one sentence 
	#    sentence = dataset["Text"][i]
	    
	    # Compute the length of the sentence
	#    length = len(sentence)
	    
	    # If the sentence length is less than 50, do padding with 50 space
	#    if length < k_num:
	#        sentence = sentence + " " * k_num
	#        length += k_num
	    
	    # Loop the sentence
	#    for j in range(length - k_num + 1):
	#        temp = sentence[j:j+k_num]
	#        if temp in k_shingles_dict:
	#            Binary_data_matrix[k_shingles_dict[temp], i] = 1





	# Question 3 
	#######################
	# design a better way to store the data by only using their non zero index
	#######################

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


	#######################
	# 10,000 random pairs of Jaccard distance
	#######################
	# 10,000 pairs 
	num_pairs = 10000
	Jaccard_distance = np.zeros((num_pairs, 1))

	print("Computing the Jaccard distance in 10,000 random paris and monitor the process")
	for i in range(num_pairs):
	    if i % 100 == 0:
	        print(i)
	    idx = np.random.choice(num_sentences, 2, replace = False)
	    sentence_1 = generate_binary_vector(idx[0])
	    sentence_2 = generate_binary_vector(idx[1])
	    
	    count_and = np.sum(sentence_1 * sentence_2)
	    count_or = np.sum(sentence_1) + np.sum(sentence_2) - count_and
	    
	    Jaccard_distance[i] = 1 - count_and / count_or
	    
	# plot the histogram
	plt.figure(1, figsize=(8,5))
	plt.hist(Jaccard_distance, label = "Jaccard_distance_histogram", bins = 100, density=True, alpha = 0.6)
	plt.xlabel('Jaccard distance',fontsize=13)
	plt.ylabel('Number of pairs',fontsize=13)
	plt.legend(loc='upper left', frameon=False, prop={'size': 10})
	plt.savefig('./10000Jaccard_distance', dpi = 600)

	# min of these Jaccard distance
	print("min of these Jaccard distance", min(Jaccard_distance))
	# mean of these Jaccard distance
	Jaccard_distance = np.asarray(Jaccard_distance)
	print("mean of these Jaccard distance", np.mean(Jaccard_distance))



	# Question 4
	#######################
	# design a better way to store the data by only using their non zero index
	# loop through all the data and for each one, construct a vector storing its non-zero indices
	# the final data set we have is a list of list (mimic sparse matrix)
	#######################


	Data_list = []
	print("Constructing a better data structur to store the data by only storing the non zero indices and monitor the process")

	# Loop through each sentence
	for i in range(num_sentences):
	    # Print the process
	    if i % 1000 == 0:
	        print(i)

	    temp_list = []
	    seen = set()
	    # Load one sentence 
	    sentence = dataset["Text"][i]
	    
	    # Compute the length of the sentence
	    length = len(sentence)
	    
	    # If the sentence length is less than 50, do padding with 50 space
	    if length < k_num:
	        sentence = sentence + " " * k_num
	        length += k_num
	    
	    # Loop the sentence
	    for j in range(length - k_num + 1):
	        temp = sentence[j:j+k_num]
	        if temp in k_shingles_dict and temp not in seen:
	            seen.add(temp)
	            temp_list.append(k_shingles_dict[temp])
	            
	    Data_list.append(temp_list)



	# Question 5
	#######################
	# Computing the smallest prime number larger than R (number of total shingles) and call this prime number R
	#######################

	# Data_list contains all the sentence and the indices of shingle they include

	# Find the smallest prime numbergreater than n_row (number of shingles) in order to make good permutations
	R = n_row

	def prime_number(num):
	   # Iterate from 2 to n / 2  
	    for i in range(2, num//2): 
	       # If num is divisible by any number between  
	       # 2 and n / 2, it is not prime  
	        if (num % i) == 0: 
	            return False 
	    else:
	        return True

	print("total number of shingles", R)
	# If given number is greater than 1 
	while prime_number(R) == False:
	    R += 1
	    
	print("Prime number we use for the min-hashing", R)

	#######################
	# constructing signature matrix
	#######################

	M = 600
	pi = np.random.randint(1, R - 1, size=(M, 2))
	print(pi.shape)


	def h_pi(data, pi, R, M):
	    data = np.asarray(data)[None,:]
	    aa = pi[:,0:1]
	    bb = pi[:,1:2]
	    hash_vector = np.min((aa * data + bb) % R  ,axis=1)[:,None]
	    return hash_vector

	Hash_matrix = np.zeros((M, num_sentences))

	print("Computing the signature matrix and monitor the process")

	for i in range(num_sentences):
	    if i % 1000 == 0:
	        print(i)
	    data = Data_list[i]
	    Hash_matrix[:, i:i+1] = h_pi(data, pi, R, M)


	#######################
	# determine the number of bands and number of elements r in each band
	#######################

	# Here we can conclude if we would like to catch all possible 80 % similar sentences, set m = 1000, r = 15, b = 40
	# If we just want to reduce the number of making mistakes 

	x = np.linspace(0, 1, 100)[:,None]

	def function_curve(x, r, b):
	    return 1 - (1 - x**r)**b

	plt.figure(10)
	for r in [5, 10, 15, 20, 40, 50, 100]:
	    y = function_curve(x, r, 600//r)
	    plt.plot(x, y, label = "r=" +str(r))
	plt.axvline(x=0.8)
	plt.axvline(x=0.5)
	    
	plt.legend(loc='upper left', frameon=False, prop={'size': 10})
	plt.xlabel('Similarity',fontsize=13)
	plt.ylabel('Pr(hit)',fontsize=13)
	plt.savefig('./Determine_M', dpi = 600)

	# set the value for r and b
	r = 10
	b = 60

	# initially set the second prime number close to 100 times the shingles size
	P = R * 100

	#print(P)
	# If given number is greater than 1 
	while prime_number(P) == False:
	    P += 1
	    
	print("prime number we use for the second min-hashing assigning the bucket values", P)

	pi_r = np.random.randint(1, P - 1, size=(r, 2))
	#print(pi_r.shape)

	#######################
	# functions for the second min-hashing
	#######################

	# function for returning the similar pairs of reviews in band # b
	def find_similar_in_band(pi_r, b, r, P):
	    buckets = {}
	    a_v = pi_r[:, 0:1]
	    b_v = pi_r[:, 1:2]
	    for i in range(num_sentences):
	        data = Hash_matrix[b*r:(b+1)*r,i:i+1]
	        temp = np.sum((data * a_v + b_v) % P)
	        if temp not in buckets:
	            buckets[temp] = [i]
	        else:
	            buckets[temp].append(i)
	            
	    return buckets #pairs
	    
	    
	def get_pairs(buckets):
	    pairs = []
	    for key in buckets:
	        length = len(buckets[key])
	        if length >= 2:
	            for i in range(length):
	                for j in range(i+1, length):
	                    pairs.append((buckets[key][i], buckets[key][j]))

	    return pairs


	#######################
	# performing the second min-hashing and get the final candidate pairs 
	#######################
	print("Do the second min-hashing and assigne values for each bands into different buckets")

	# Store all the buckets information for all the bands
	list_buckets = []

	# Compute the bucket for the first band
	buckets = find_similar_in_band(pi_r, 0, r, P)

	# Append the computed bucket in the list
	list_buckets.append(buckets)

	# Parse the bucket information to get final pairs
	band_pair = get_pairs(buckets)

	# Build a dictionary of final candidate pairs (to avoid repeat count)
	final_pair = set()
	for i in range(len(band_pair)):
	    if band_pair[i] not in final_pair:
	        final_pair.add(band_pair[i])

	print("Going through the data band by band with total number of band", b)

	# Compute the bucket for all the rest bands and parse the data
	print(len(final_pair))
	for i in range(1, b):
	    print(i)
	    buckets = find_similar_in_band(pi_r, i, r, P)
	    list_buckets.append(buckets)
	    get = get_pairs(buckets)
	    for j in range(len(get)):
	        if get[j] not in final_pair:
	            final_pair.add(get[j])

	print("number of final candidate pairs", len(final_pair))
	    
	#######################
	# Looping through the candidate pairs and compute the Jaccard distance for determining the similarity
	#######################

	final_length = len(final_pair)


	output_pair = []
	count = 0

	# Total Jaccard distance
	Jaccard_distance_new_1 = []

	# Accepted Jaccard distance
	Jaccard_distance_new_2 = []

	print("Going through all the candidate pairs and check their Jaccard distance to determine whether to accept or not")

	for x in final_pair:
	    count += 1
	    if count % 100 == 0:
	        print(count)
	    idx1, idx2 = x
	    sentence_1 = generate_binary_vector(idx1)
	    sentence_2 = generate_binary_vector(idx2)
	    
	    count_and = np.sum(sentence_1 * sentence_2)
	    count_or = np.sum(sentence_1) + np.sum(sentence_2) - count_and
	    
	    Jaccard_distance = 1 - count_and / count_or
	    Jaccard_distance_new_1.append(Jaccard_distance)
	    
	    if Jaccard_distance <= 0.2:
	        output_pair.append([idx1, idx2])
	        Jaccard_distance_new_2.append(Jaccard_distance)

	plt.figure(3, figsize=(8,6))
	plt.hist(Jaccard_distance_new_2, label = "Accepted Jaccard_distance", bins = 100, density=True, alpha = 0.6)
	plt.legend(loc='upper right', frameon=False, prop={'size': 10})
	plt.savefig('./Accepted_Jaccard_Distance', dpi = 600)
	
	print("final number of similar paris in the data set", len(output_pair))

	# store all the pairs into csv file
	import csv

	num_final_output_paris = len(output_pair)

	w = csv.writer(open("close_pairs.csv", "w"))
	for i in range(num_final_output_paris):
	    idx_sim_1, idx_sim_2 = output_pair[i]
	    w.writerow([dataset["ID"][idx_sim_1], dataset["ID"][idx_sim_2]])


	# Question 6
	#######################
	# finding the nearest reviewID given a new queried review 
	#######################




	# relax the similarity to be 0.2 for the later work
	r = 2
	b = 300

	pi_r = np.random.randint(1, P - 1, size=(r, 2))

	print("Building a relax threshold for finding nearest neighbor that has at least similarity 0.5")

	# Store all the buckets information for all the bands
	list_buckets = []

	# Compute the bucket for the first band
	buckets = find_similar_in_band(pi_r, 0, r, P)

	# Append the computed bucket in the list
	list_buckets.append(buckets)


	print("Going through the data band by band for the relaxed case, with total number of band", b)

	# Compute the bucket for all the rest bands and parse the data
	print(len(final_pair))
	for i in range(1, b):
	    print(i)
	    buckets = find_similar_in_band(pi_r, i, r, P)
	    list_buckets.append(buckets)


	######### store all the datas needed from prediction 

	
	np.save("pi_r", pi_r)
	np.save("pi", pi)
	np.save("P", P)
	np.save("R", R)


	w = csv.writer(open("shingle_dict.csv", "w"))
	for key, val in k_shingles_dict.items():
		w.writerow([key, val])


	num_buckets = len(list_buckets)

	import json
	
	with open("list_buckets.json", "w") as f:
		json.dump(list_buckets, f)


	# generate vector storing the non zero indices for a given sentence 
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


	#######################
	# function for finding the nearest reviewID given a new queried review 
	#######################

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
	            Jaccard_distance_new_1.append(Jaccard_distance)

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




