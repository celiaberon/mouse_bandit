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
from sklearn.model_selection import train_test_split


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

def feature_history(data, X_dataframe='switch', y_dataframe='switch'):
    #include past 10 choice and reward values (this is most convenient given the current data structure)
    port_features = []
    reward_features = []
    for col in data.columns:
        if '_Port' in col:
            port_features.append(col)
        if '_Reward' in col:
            reward_features.append(col)

    choice_raw = data[port_features] # choice history
    c = choice_raw.copy()
    c[choice_raw==0]=-1 # make right choice -1 instead of zero (important for making final dataframe)
    r = data[reward_features] # reward history
    
    switch_cols = ['9_Switch','8_Switch','7_Switch','6_Switch','5_Switch','4_Switch','3_Switch','2_Switch','1_Switch']
    switch_raw = pd.DataFrame(np.abs(np.diff((c))))
    switch_raw.columns=switch_cols
    switch_raw.index=r.index
    s = switch_raw.copy()
    s[switch_raw==0]=-1 # same thing as for choice, make "stays" -1 instead of 0
    
    if X_dataframe == 'choice':
        #X = pd.concat([choice_history.drop('10_Port', axis=1), reward_history.drop('10_Reward', axis=1)], axis=1)
        X = c.values * r.values # now only have 1 where left choice was rewarded, -1 where right choice was rewarded
    elif X_dataframe == 'switch':
        #X = pd.concat([switch_history, r.drop('10_Reward', axis=1)], axis=1)
        r = r.drop('10_Reward', axis=1)
        X = s.values * r.values
    elif X_dataframe == 'choice_switch':
        X = pd.concat([switch_history, c.drop('10_Port', axis=1), r.drop('10_Reward', axis=1)], axis=1)
    elif X_dataframe == 'value':
        X = (c.values==r.values).astype('int') # gives action value with R=1, L=0

    if y_dataframe == 'switch':
        y_raw = data['Switch']
        y = y_raw.copy()
        y[y_raw==0]=-1
    elif y_dataframe == 'choice':
        y_raw = data['Decision']
        y=y_raw.copy()
        y[y_raw==0]=-1
        
    return X, y, c

def choice_history_lateral(data):
    #include past 10 choice and reward values (this is most convenient given the current data structure)
    port_features = []
    reward_features = []
    for col in data.columns:
        if '_Port' in col:
            port_features.append(col)
        if '_Reward' in col:
            reward_features.append(col)

    choice_raw = data[port_features] # choice history
    reward_raw = data[reward_features] # reward history
    y_raw = data['Decision']
    
    c = choice_raw.copy()
    r = reward_raw.copy()

   
    r[reward_raw==0]=-1 # make unrewarded -1
    
    '''left choice dataframes'''
    # right choice already equals 0 in original data
    X_left = c.values * r.values # 1=left rewarded, -1=left unrewarded, 0=right
    y_left=y_raw.copy()

    
    '''right choice dataframes'''
    c[choice_raw==0]=1 # make right choice 1 instead of zero (basically flipping binary rep to match with left dataframe)
    c[choice_raw==1]=0
    
    X_right = c.values *r.values
   
    y_right=y_raw.copy()
    y_right[y_raw==0]=1
    y_right[y_raw==1]=0
                    
    return X_left, X_right, y_left, y_right, c

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

def logreg_predict_proba_metrics(X, y, c, n_simulations=10, test=0.3, sync=False, seed=[], choice=False):
    # select seed for each simulation
    if len(seed)!=n_simulations:
        seed = np.random.randint(100000, size=n_simulations)

    # initialize arrays for LR results
    lr_predict = np.zeros((int(np.round(test*len(X))),n_simulations))
    lr_proba = np.zeros((int(np.round(test*len(X))), 2))
    lr_score = np.zeros(n_simulations)
    metrics = np.zeros((4,2))

    for i in range(n_simulations):
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=test, random_state=seed[i])

        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        lr_proba = np.dstack((lr_proba, lr.predict_proba(X_test)))
        lr_predict[:,i] = lr.predict(X_test)
        lr_score[i] = lr.score(X_test, y_test)
        metrics_temp = sklearn.metrics.precision_recall_fscore_support(y_test, lr_predict[:,i])
        metrics = np.dstack((metrics, np.array(metrics_temp)))
        
        if choice==True:
            metrics_switch = np.zeros((4,2))
            switch_predict = np.zeros((int(np.round(test*len(X))),n_simulations))
            
            prev_choice = c_test['1_Port']
            switch_true = np.abs(y_test-prev_choice)
            switch_predict[:,i] = np.abs(lr_predict[:,i]-prev_choice)
            metrics_switch_temp = sklearn.metrics.precision_recall_fscore_support(switch_true, switch_predict[:,i])
            metrics_switch = np.dstack((metrics_switch, np.array(metrics_switch_temp)))

    # clean up from initializing 3d stacks with zeros
    metrics = metrics[:,:,1:n_simulations+1]
    lr_proba = lr_proba[:,:,1:n_simulations+1]
        
    if sync==True: 
        return metrics, lr_predict, lr_proba, lr_score, seed
    elif choice==True:
        metrics_switch = metrics_switch[:,:,1:n_simulations+1]

        return metrics_switch, lr_predict, lr_proba, lr_score
    else:
        return metrics, lr_predict, lr_proba, lr_score


def logreg_sim_plot(data, n_simulations=10, test=0.3, sync=False, seed=[], choice=False):

    # prep data for simulation
    seed1 = np.random.randint(100000,size=n_simulations)

    X_9010, y_9010, c_9010 = feature_history(data[data['Condition']=='90-10'], 
                                                X_dataframe='choice', y_dataframe='choice')
    metrics9010, lr_predict9010, lr_proba9010, lr_score9010 = logreg_predict_proba_metrics(X_9010, 
                                                y_9010, c_9010, n_simulations=n_simulations, seed=seed1, choice=True)

    X_8020, y_8020, c_8020 = feature_history(data[data['Condition']=='80-20'], 
                                                X_dataframe='choice', y_dataframe='choice')
    metrics8020, lr_predict8020, lr_proba8020, lr_score8020 = logreg_predict_proba_metrics(X_8020, 
                                                y_8020, c_8020, n_simulations=n_simulations, seed=seed1, choice=True)

    X_7030, y_7030, c_7030 = feature_history(data[data['Condition']=='70-30'], 
                                                X_dataframe='choice', y_dataframe='choice')
    metrics7030, lr_predict7030, lr_proba7030, lr_score7030 = logreg_predict_proba_metrics(X_7030, 
                                                y_7030, c_7030, n_simulations=n_simulations, seed=seed1, choice=True)

    # make bar plot of data

    height_9010 = [np.mean(lr_score9010), np.mean(metrics9010[1,0,:]), np.mean(metrics9010[1,1,:])]
    ystd1 = [np.std(lr_score9010), np.std(metrics9010[1,0,:]), np.std(metrics9010[1,1,:])]
    yerr1 = [ystd1[i] / np.sqrt(n_simulations) for i in range(len(ystd1))]

    height_8020 = [np.mean(lr_score8020), np.mean(metrics8020[1,0,:]), np.mean(metrics8020[1,1,:])]
    ystd2 = [np.std(lr_score8020), np.std(metrics8020[1,0,:]), np.std(metrics8020[1,1,:])]
    yerr2 = [ystd2[i] / np.sqrt(n_simulations) for i in range(len(ystd2))]

    height_7030 = [np.mean(lr_score7030), np.mean(metrics7030[1,0,:]), np.mean(metrics7030[1,1,:])]
    ystd3 = [np.std(lr_score7030), np.std(metrics7030[1,0,:]), np.std(metrics7030[1,1,:])]
    yerr3 = [ystd3[i] / np.sqrt(n_simulations) for i in range(len(ystd3))]

    barWidth = 0.2
    # The x position of bars
    r1 = np.arange(len(height_9010))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    conditions = ['full', 'stay', 'switch']
    plt.bar(r1, height_9010, width=barWidth, label='9010', yerr=yerr1, capsize=3)
    plt.bar(r2, height_8020, width=barWidth, label='8020', yerr=yerr2, capsize=3)
    plt.bar(r3, height_7030, width=barWidth, label='7030', yerr=yerr3, capsize=3)

    plt.xticks(range(len(height_9010)), conditions)
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

