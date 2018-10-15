#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:14:43 2018

@author: celiaberon
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn
import itertools
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline

def lr_bandit_sim(X, y, n_simulations=1000, cutoff=0.7, sync=False):

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

def Xy_history(data, X_dataframe='switch', y_dataframe='switch'):
    #include past 10 choice and reward values (this is most convenient given the current data structure)
    port_features = []
    reward_features = []
    for col in data.columns:
        if '_Port' in col:
            port_features.append(col)
        if '_Reward' in col:
            reward_features.append(col)

    choice_history = data[port_features]
    reward_history = data[reward_features]

    switch_cols = ['9_Switch','8_Switch','7_Switch','6_Switch','5_Switch','4_Switch','3_Switch','2_Switch','1_Switch']
    switch_history = pd.DataFrame(np.abs(np.diff((choice_history))))
    switch_history.columns=switch_cols
    switch_history.index=reward_history.index

    if X_dataframe == 'choice':
        X = pd.concat([choice_history.drop('10_Port', axis=1), reward_history.drop('10_Reward', axis=1)], axis=1)
    elif X_dataframe == 'switch':
        X = pd.concat([switch_history, reward_history.drop('10_Reward', axis=1)], axis=1)
    elif X_dataframe == 'choice_switch':
        X = pd.concat([switch_history, choice_history.drop('10_Port', axis=1), reward_history.drop('10_Reward', axis=1)], axis=1)
    elif X_dataframe == 'value':
        X = (choice_history.values==reward_history.values).astype('int') # gives action value with R=1, L=0

    if y_dataframe == 'switch':
        y = data['Switch']
    elif y_dataframe == 'choice':
        y = data['Decision']
        
    return X, y

def sequences_predict_switch(X, y, sequence_length=3, display=True):
    
    switch_reward_mat = X[['3_Switch','2_Switch','1_Switch','3_Reward','2_Reward','1_Reward']]
    reward_combos = list(itertools.product([0,1], repeat=sequence_length*2, ))
    
    switch_reward_sequence = []
    for i in range(len(X)):
        switch_reward_sequence.append(np.where(np.sum(reward_combos==switch_reward_mat.values[i], axis=1)==6)[0][0])
        
    idx=[] # create vector with indices for where each sequence in reward combos occurs in test
    for i in range(len(reward_combos)):
        idx.append(np.where([switch_reward_sequence[n]==i for n in range(len(switch_reward_sequence))])[0])
        
    prob_switch_counts = np.array([len(idx[i]) for i in range(len(reward_combos))])

    prob_switch=np.zeros(len(reward_combos))
    for i in range(len(reward_combos)):
        if prob_switch_counts[i]==0:
            prob_switch[i] = 0
        else:
            prob_switch[i] = y.iloc[idx[i]].sum()/len(idx[i]) 
    
    if display==True:
        prob_switch=np.array(prob_switch)
        idx = prob_switch.argsort()
        reward_combos_sorted=np.array(reward_combos)[idx]
        prob_switch_sorted=prob_switch[idx]
        prob_switch_counts_sorted=prob_switch_counts[idx]

        reward_intensity = [reward_combos_sorted[i][3:6].sum() for i in range(len(reward_combos_sorted))]
        colors_dict={0:'whitesmoke', 1:'silver', 2:'dimgray', 3:'black'}
        colors = [colors_dict[reward_intensity[i]] for i in range(len(reward_intensity))]

        plt.figure(figsize=(20,8))
        plt.bar(left=np.arange(len(reward_combos)) ,height=prob_switch_sorted, color=colors)
        plt.xticks(range(len(reward_combos)), reward_combos_sorted, rotation='vertical')
        plt.ylabel('p_switch')
        plt.xlabel('sequence - switch, reward')
        print()
        plt.legend(colors_dict, loc='upper left')

        for i in range(len(reward_combos)):
            plt.text(x = np.arange(len(reward_combos))[i]-.4 , y = prob_switch_sorted[i]+0.01, 
             s = "{:.1f}".format((prob_switch_counts_sorted[i]/prob_switch_counts.sum())*100), size = 8)
      
    return reward_combos, prob_switch    
        