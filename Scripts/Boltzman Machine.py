#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################
"""
    Author: xhuang
    Date created: 4/20/2019
    Date last modified: 4/25/2019
    Python Version: 3.7
"""
#############################################
# Import necessary library
import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
import os

os.chdir("D:\Data Science\DeepLearningXoanon\Scripts")
print(os.getcwd())

movie_list = pd.read_csv("../Data/BM/ml-1m/movies.dat",sep='::', header=None, engine='python', encoding='latin-1')
user_list = pd.read_csv("../Data/BM/ml-1m/users.dat",sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv("../Data/BM/ml-1m/ratings.dat",sep='::', header=None, engine='python', encoding='latin-1')


training_set = np.array(pd.read_csv("../Data/BM/ml-100k/u1.base", delimiter='\t'), dtype='int')
test_set = np.array(pd.read_csv("../Data/BM/ml-100k/u1.test", delimiter='\t'), dtype='int')

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

def convert(data):
    data_pivoted = []
    for id_user in range(1,nb_users+1):
        data_users = data[data[:,0] == id_user]
        rating = np.zeros(nb_movies)
        rating[data_users[:,1] - 1] = data_users[:,2]
        data_pivoted.append(rating)
    return np.array(data_pivoted)

training_set = convert(training_set)
test_set = convert(test_set)


# Converting into torch tensors 
training_set = tc.FloatTensor(training_set)
test_set = tc.FloatTensor(test_set)

# Converting to Binary rating  1 like 0 dislike
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating NN RBM
class RBM():
    def __init__(self, nv, nh):
        self.W = tc.randn(nh, nv)
        self.a = tc.randn(1, nh)
        self.b = tc.randn(1, nv)

    def sample_h(self, x):
        wx = tc.mm(x, self.W.t())
        activiation = wx + self.a.expand_as(wx)
        p_h_given_v = tc.sigmoid(activiation)
        return p_h_given_v, tc.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = tc.mm(y, self.W) 
        activiation = wy + self.b.expand_as(wy)
        p_v_given_h = tc.sigmoid(activiation)
        return p_v_given_h, tc.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk): 
        self.W += (tc.mm(v0.t(), ph0) - tc.mm(vk.t(), phk)).t()
        self.b += tc.sum((v0 - vk), 0)
        self.a += tc.sum((ph0 - phk), 0)
    
nv = nb_movies
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        # Initialize parametres
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        
        # Constractive Divergence with k steps
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += tc.mean(tc.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1
    print('epoch:{}'.format(epoch))
    print('loss: {}, s:{}'.format(train_loss, s))
    
    
    
# Test 
test_loss = 0
test_s = 0
for id_user in range(nb_users):  
    # Use training set to activate neurons
    v = training_set[id_user:id_user+1]
    # Use test to get the predicted rating
    vt = test_set[id_user:id_user+1]
    
    # One step gibbs sampling. 
    # only when Users in testset has at least one rating, we can compare the predicted rating with the truth
    if len(vt[vt>=0]) > 0 :
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        
        test_loss += tc.mean(tc.abs(vt[vt>=0] - v[vt>=0]))
        test_s += 1   
print('test_loss: {}, test_s:{}'.format(test_loss, test_s))
    



