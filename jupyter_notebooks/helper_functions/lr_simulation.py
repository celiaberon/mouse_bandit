#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:14:43 2018

@author: celiaberon
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn

def lr_bandit(X, y, n_simulations=1000, cutoff=0.7, sync=False):

    # trim down datasets to be divisible by 10 (temporary hack ok with large dataset)
    X = X[0:-np.remainder(len(X),10)]
    y = y[0:-np.remainder(len(y),10)]
    
    # select seed for each simulation
    seed = np.random.randint(100000, size=n_simulations)

    # initialize arrays for LR results
    lr_predict = np.zeros((int((1-cutoff)*len(X)),n_simulations))
    lr_proba = np.zeros((int((1-cutoff)*len(X)), 2))
    lr_score = np.zeros(n_simulations)
    metrics = np.zeros((4,2))

    for i in range(n_simulations):
        
        np.random.seed(seed[i])
        idx = np.random.permutation(len(X))
        idx_train = idx[0:int(len(X)*cutoff)]
        idx_test = idx[int(len(X)*cutoff):]

        X_train = X[idx_train]
        X_test = X[idx_test]

        Y_train = y[idx_train]
        Y_test = y[idx_test]

        lr = LogisticRegression()
        lr.fit(X_train, Y_train)

        lr_proba = np.dstack((lr_proba, lr.predict_proba(X_test)))
        lr_predict[:,i] = lr.predict(X_test)
        lr_score[i] = lr.score(X_test, Y_test)
        metrics_temp = sklearn.metrics.precision_recall_fscore_support(Y_test, lr_predict[:,i])
        metrics = np.dstack((metrics, np.array(metrics_temp)))
        
        #lr1_switch_predict = np.abs([lr1_choice_predict[n] - X_test[n,9] for n in range(len(lr1_choice_predict))])
        #lr1_metrics_switch = sklearn.metrics.precision_recall_fscore_support(Y_test_switch, lr1_switch_predict[:,i])

    # clean up from initializing 3d stacks with zeros
    metrics = metrics[:,:,1:n_simulations+1]
    lr_proba = lr_proba[:,:,1:n_simulations+1]
    
    if sync==True: 
        return metrics, lr_predict, lr_proba, lr_score, seed
    return metrics, lr_predict, lr_proba, lr_score