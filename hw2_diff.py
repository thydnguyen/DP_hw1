# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:31:46 2021

@author: Thy Nguyen
"""

import numpy as np

n_range = [50, 100, 500, 2000, 10000]
GaussR = [100, 1000, 10000]
PoissonR = [100, 1000, 10000]
BiK = [10, 100, 200]
dataset = [GaussR, PoissonR, BiK]
names = ["gauss", "poisson", "bi"]
numDataSet = 50
repeat = 10

#Synthetic data generator
def syntheticData(n, TYPE, param):
    if TYPE == "gauss":
        return np.clip(np.round(np.random.normal(param / 4, param**2 / 10, n)), 1, param)
    elif TYPE  == "poisson":
        return np.clip(np.round(np.random.poisson(50, n)), 1 , param)
    elif TYPE == "bi":
        return np.clip(np.round(np.random.choice([500 - param, 500 + param],n )), 1, 1000)
    
def median(data, R):
    data_sort = np.sort(data).tolist()
    n = len(data)
    data_unique, count = np.unique(data , return_counts = True)
    data_dict = {}
    for x, c in zip (data_unique, count):
        data_dict[x] = c
    starting_pt = 1
    score_list = [0] * R
    #we iterate the sorted data, we assign the rank of each i in [1,R] by when we find x_i great than i
    #When we find such x_i, we iterate from i to R, until we find one that is bigger than x_i,
    # We assign rank value for other values in the loop that are not greater than x_i
    #This takes O(n) time
    for (pos, j) in enumerate(data_sort):
        if starting_pt > j:
            continue
        else:
            score_list[starting_pt] = pos
            for j_rest in range(starting_pt + 1, R + 1):
                if j_rest < j:
                    score_list[j_rest] = pos
                else:
                    starting_pt = j_rest
                    break
            starting_pt = j_rest
        if starting_pt == R:
            break
    for j in range(starting_pt, R):
        score_list[j] = n
    
    #We have counts of every unique element in data. We can use that and the calculated rank 
    #to get the score for each value 1 to R     
    rank = score_list[:]
    for j in range(R):
        if j in data_dict.keys():
            total =  n - data_dict[j]
        else:
            total = n
        score_list[j] = abs(total - 2 * score_list[j])
    
    returnList = []
    for i in range(repeat):
        pertubed_score = [y + np.random.exponential(20) for y in score_list]
        returnList.append(abs(rank[np.argmin(pertubed_score)] - n/2))
    return returnList


result = {}
for d, name in zip(dataset, names):
    result[name] =  np.zeros((len(d), len(n_range))).tolist() 
    for  n_id, n in enumerate(n_range):
        for param_id, param in enumerate(d):
            allScores = []
            allStd = []
            for d_i in range(numDataSet):
                if name != "bi":
                    scores = median(syntheticData(n, name, param), param)
                else:
                    scores = median(syntheticData(n, name, param), 1000)
                allScores.extend(scores)
                allStd.append(np.std(scores))
            mean = str(np.round(np.mean(allScores), 1))
            stdev_pop = str(np.round(np.std(allScores), 1))
            stdev_whole = str(np.round(sum(allStd) / len(allStd), 1))
            result[name][param_id][n_id] = mean + "|" + stdev_pop + "|" + stdev_whole

    
        
        

    
    
    
