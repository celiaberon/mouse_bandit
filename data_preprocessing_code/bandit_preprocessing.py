"""
Created on Wed Nov 16 20:32:48 2016

@author: shayneufeld

This file contains preprocessing functions to create feature dataframes
for our AC209A project modeling the 2-armed bandit task in mice
"""
import numpy as np
import pandas as pd
from scipy import optimize

def create_feature_matrix(trials,n_indi,mouse_id,session_id,
                          feature_names='Default',curr_trial_duration=True,
                          imaging=False):
    '''
    This function creates the feature matrix we will use!
    
    Inputs:
            trials       :  pandas dataframe returned by extractTrials.m
            n_indi       :  number of past trials to be used in individual trial features
            mouse_id     : mouse id
            session_id   : sesion_id
            feature_names: list of column names for the dataframe
            curr_trial_duration: true/false whether to include
            imaging      : true/false whether is imaging data for this behavior session
    Outputs:
            feature_trials: pandas dataframe of the features for each trial
    
    Note:
        - this only considers trials 10 to the end
        - it assumes n_summary > n_indi
        
    '''
    n_trials = trials.shape[0] #number of trials in this session
    
    #minimum number of columns
    num_cols = 2 + 3+ 4*n_indi+ 1 + 4
    #2 streak cols, 3 trial & block & reward trial col, 4 cols for each past trial, 
    #]1 for current trial + 4 decision/switch/higher p port/Reward
    
    #add columns as neccesary
    if curr_trial_duration is True:
        num_cols += 1
    if imaging is True:
        num_cols += 2
    
    feature_matrix = np.zeros((n_trials-n_indi,num_cols))
    
    block_starts = np.zeros(trials.shape[0])
    block_starts[1:] = np.diff(trials['Right Reward Prob'].values) != 0
    
    reward_block_num = 0    
    reward_block_nums = np.zeros(n_trials)
    for i in range(n_trials):
        if block_starts[i]:
            reward_block_num = 0
        else:
            reward_block_num += trials.iloc[i]['Reward Given']
    
        reward_block_nums[i] = reward_block_num

    for j,i in enumerate(np.arange(n_indi,n_trials)):
        k = 0 #column indexer for the feature matrix
        #extract the 'n_summary' trials we need to consider. Assume that n_summary > n_indi
        past_trials = trials.iloc[i-n_indi:i]
        
        '''
        Mouse ID
        '''
        # will be added after (since its a string)
        
        '''
        Session ID
        '''
        # will be added after (since its a string)
        
        '''
        Trial Number
        '''
        feature_matrix[j,k] = i+1
        k+=1 
        # will be added after all at once (since can be calculated for every trial with single line)
        
        '''
        Block Trial Number
        '''
        if (j == 0): #first block number will be the sum
            if (np.sum(block_starts[:n_indi]) == 0):
            #then we are still in the first block, and we simply are n_indi trials in
                feature_matrix[j,k] = n_indi+1
            else:
                feature_matrix[j,k] = n_indi - np.where(block_starts[:n_indi]==True)[0][0] + 1
                # i.e. 10 - 5 + 1 = 6 or 10 - 8 + 1 = 3. it works. 
        elif block_starts[i]: #if block_starts[j] == True, start counting from 0
            feature_matrix[j,k] = 0 
        else:
            feature_matrix[j,k] = feature_matrix[j-1,k] + 1
            
        k+=1
        
        '''
        Block Reward Number
        '''
        feature_matrix[j,k] = reward_block_nums[i-1]
        k+=1
            
        
        '''
        PORT STREAK
        
        approach: take the derivative of the 'port poked' variabe. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakP_vec = np.flipud(np.diff(trials['Port Poked'].values[:i])) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakP_len = np.nonzero(streakP_vec)[0]

        if i == 0:
            feature_matrix[j,k] = 0 #special case on first trial
        elif len(streakP_len) == 0: #have to deal with case where streak is all 10 previous trials!
            feature_matrix[j,k] = i
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakP_len = streakP_len[0]
            feature_matrix[j,k] = streakP_len+1
        k+=1
        
        '''
        REWARD STREAK
        
        approach: take the derivative of the reward boolean. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakR_vec = np.flipud(np.diff(trials['Reward Given'].values[:i])) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakR_len = np.nonzero(streakR_vec)[0]
        
        if i == 0:
            streakR_len = 0 
            streakR_sign = 0 
            #special case on first trial, reward streak should be 0! 
            # we can trivially set this to -1 to make that the case below. 
            
        elif len(streakR_len) == 0: #have to deal with case where streak is all 10 previous trials!
            streakR_len = i-1
            if np.sum(trials['Reward Given'].values[:i] > 0):
                streakR_sign = 1

            else:
                streakR_sign = -1
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakR_len = streakR_len[0]
            streakR_sign = streakR_vec[streakR_len]
        
        feature_matrix[j,k] = (streakR_len+1)*streakR_sign
            
        k+=1
    
        
        '''
        INDIVIDUAL TRIALS
        '''
        for icol,itrial in enumerate(np.arange(n_indi,0,-1)):
            
            past_trial = past_trials.iloc[-itrial,:]
            
            #which port
            if past_trial['Port Poked'] == 1:
                feature_matrix[j,k] = 0
            elif past_trial['Port Poked'] == 2:
                feature_matrix[j,k] = 1
            else:
                print('Error port not Left or Right')
            k += 1
            
            #reward given:
            feature_matrix[j,k] = past_trial['Reward Given']
            k += 1
            
            #ITI
            feature_matrix[j,k] = past_trial['Since last trial (s)']
            k += 1
            
            #trial time
            feature_matrix[j,k] = past_trial['Trial Duration (s)']
            k += 1
        
        '''
        CURRENT TRIAL
        '''
        current_trial = trials.iloc[i,:]
        feature_matrix[j,k] = current_trial['Since last trial (s)']
        k += 1
        
        '''
        Add in current trial duration (if flag is set to true)
        '''
        if curr_trial_duration == True:
            feature_matrix[j,k] = trials.iloc[i]['Trial Duration (s)']
            k+=1
    
        '''
        DECISION
        '''
        if current_trial['Port Poked'] == 1:
            feature_matrix[j,k] = 0
        elif current_trial['Port Poked'] == 2:
            feature_matrix[j,k] = 1
        else:
            print('Error decision port not Left or Right')
        
        k += 1
        '''
        SWITCH
        '''
        feature_matrix[j,k] = np.abs((current_trial['Port Poked'] - trials.iloc[i-1]['Port Poked']))
        
        k+= 1
        
        '''
        p(high) port
        '''
        
        if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
            feature_matrix[j,k] = 1
        elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
            feature_matrix[j,k] = 1
        else:
            feature_matrix[j,k] = 0
        
        k+=1
        
        '''
        REWARD
        '''
        feature_matrix[j,k] = current_trial['Reward Given']
        k+=1
        
        '''
        sync to imaging data
        '''
        if imaging is True:
            feature_matrix[j,k] = current_trial['center_frame']
            k += 1
            feature_matrix[j,k] = current_trial['decision_frame']
        
        
        
        
    d = {'Mouse ID':mouse_id,'Session ID':session_id}
    feature_trials = pd.DataFrame(data=d,index=range(feature_matrix.shape[0]))
    
    if feature_names == 'Default':

        feature_names = [
                    'Trial',
                    'Block Trial',
                    'Block Reward',
                    'Port Streak',
                    'Reward Streak',
                    '10_Port','10_Reward','10_ITI','10_trialDuration',
                    '9_Port','9_Reward','9_ITI','9_trialDuration',
                    '8_Port','8_Reward','8_ITI','8_trialDuration',
                    '7_Port','7_Reward','7_ITI','7_trialDuration',
                    '6_Port','6_Reward','6_ITI','6_trialDuration',
                    '5_Port','5_Reward','5_ITI','5_trialDuration',
                    '4_Port','4_Reward','4_ITI','4_trialDuration',
                    '3_Port','3_Reward','3_ITI','3_trialDuration',
                    '2_Port','2_Reward','2_ITI','2_trialDuration',
                    '1_Port','1_Reward','1_ITI','1_trialDuration',
                    '0_ITI',
                    'Decision',
                    'Switch',
                    'Higher p port',
                    'Reward'
                     ]
                     
        if curr_trial_duration is True:
            feature_names.insert(46,'0_trialDuration')
        if imaging is True:
            feature_names.append('center_frame')
            feature_names.append('decision_frame')
            
    
    feature_trials = pd.concat([feature_trials,pd.DataFrame(data=feature_matrix,index=None,columns=feature_names)],axis=1)
    
    return feature_trials
    
def create_reduced_feature_matrix(trials,mouse_id,session_id,feature_names='Default',curr_trial_duration=False):
    '''
    This function creates the feature matrix we will use!
    
    Inputs:
            trials       :  pandas dataframe returned by extractTrials.m
            mouse_id     : mouse id
            session_id   : sesion_id
            feature_names: list of column names for the dataframe
    Outputs:
            feature_trials: pandas dataframe of the features for each trial
    
    Note:
        - this only considers trials 10 to the end
        - it assumes n_summary > n_indi
        
    '''
    n_trials = trials.shape[0] #number of trials in this session
    
    if curr_trial_duration is True:
        num_cols = 8
    else:
        num_cols = 7
    
    feature_matrix = np.zeros((n_trials,num_cols))
    
    block_starts = np.zeros(trials.shape[0])
    block_starts[0] = 1
    block_starts[1:] = np.diff(trials['Right Reward Prob'].values) != 0
                           
    for i in range(n_trials):
        k = 0 #iterate over columns
        '''
        Mouse ID
        '''
        # will be added after (since its a string)
        
        '''
        Session ID
        '''
        # will be added after (since its a string)
        
        '''
        Block Number
        '''
        # will be added after all at once (since can be calculated for every trial with single line)
        
        '''
        Block Trial Number
        '''
        if block_starts[i]: #if block_starts[j] == True, start counting from 0
            feature_matrix[i,k] = 0 
        else:
            feature_matrix[i,k] = feature_matrix[i-1,0] + 1
        k+=1
            
        '''
        PORT STREAK
        
        approach: take the derivative of the 'port poked' variabe. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakP_vec = np.flipud(np.diff(trials['Port Poked'].values[:i])) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakP_len = np.nonzero(streakP_vec)[0]
        
        if i == 0:
            feature_matrix[i,k] = 0 #special case on first trial
        elif len(streakP_len) == 0: #have to deal with case where streak is all 10 previous trials!
            feature_matrix[i,k] = i
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakP_len = streakP_len[0]
            feature_matrix[i,k] = streakP_len+1
        k+=1
        
        '''
        REWARD STREAK
        
        approach: take the derivative of the reward boolean. Streak is number of [0s + 1] (from the end). 
        the valence of the streak is the sign of first non-zero entry (from the end)
        '''
        streakR_vec = np.flipud(np.diff(trials['Reward Given'].values[:i])) #reverse order of array so end is
        #at the front. This makes it easier to find the first non-zero entry
        streakR_len = np.nonzero(streakR_vec)[0]
        
        if i == 0:
            streakR_len = 0 
            streakR_sign = 0 
            #special case on first trial, reward streak should be 0! 
            # we can trivially set this to -1 to make that the case below. 
            
        elif len(streakR_len) == 0: #have to deal with case where streak is all 10 previous trials!
            streakR_len = i-1
            if np.sum(trials['Reward Given'].values[:i] > 0):
                streakR_sign = 1

            else:
                streakR_sign = -1
        #otherwise, streak is less then 10 trials and things are simpler.
        else:
            streakR_len = streakR_len[0]
            streakR_sign = streakR_vec[streakR_len]
        
        feature_matrix[i,k] = (streakR_len+1)*streakR_sign
            
        k+=1
        
        if curr_trial_duration == True:
            feature_matrix[i,k] = trials.iloc[i]['Trial Duration (s)']
            k+=1
    
        '''
        DECISION
        '''
        current_trial = trials.iloc[i]
        
        if current_trial['Port Poked'] == 1:
            feature_matrix[i,k] = 0
        elif current_trial['Port Poked'] == 2:
            feature_matrix[i,k] = 1
        else:
            print('Error decision port not Left or Right')
        
        k += 1
        '''
        SWITCH
        '''
        feature_matrix[i,k] = np.abs((current_trial['Port Poked'] - trials.iloc[i-1]['Port Poked']))
        
        k+= 1
        
        '''
        p(high) port
        '''
        
        if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
            feature_matrix[i,k] = 1
        elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
            feature_matrix[i,k] = 1
        else:
            feature_matrix[i,k] = 0
        
        k+=1
        
        '''
        REWARD
        '''
        
        feature_matrix[i,k] = current_trial['Reward Given']
        
        
        
    d = {'Mouse ID':mouse_id,'Session ID':session_id}
    feature_trials = pd.DataFrame(data=d,index=range(feature_matrix.shape[0]))
    
    if feature_names == 'Default':
        if curr_trial_duration is True:
            feature_names = [
                        'Block Trial',
                        'Port Streak',
                        'Reward Streak',
                        'Trial Duration',
                        'Decision',
                        'Switch',
                        'Higher p port',
                        'Reward'
                         ]
        else:
            feature_names = [
                            'Block Trial',
                            'Port Streak',
                            'Reward Streak',
                            'Decision',
                            'Switch',
                            'Higher p port',
                            'Reward'
                             ]
    
    feature_trials = pd.concat([feature_trials,pd.DataFrame(data=feature_matrix,index=None,columns=feature_names)],axis=1)
    
    return feature_trials
    

def extract_session_stats(data):
    '''
    Inputs:
        data - (pandas dataframe) feature matrix (reduced or not)
    
    Outputs:
        dataframe with 3 columns:
            stable_phigh- prob/rate of choosing high p port at end of block (last 10 trials)
            peak_pswitch- prob/rate of switching at beginning of block (first 10 trials)
            rebias_tau-   time constant of exponential function fit to p(high p port) after block switch
    '''
    
    
    #all the block numbers
    t_block_unique = np.unique(data['Block Trial'].values)

    # initialize matrix for p(switch) at every trial number in block. 2nd column for SEM
    p_switch_block = np.zeros((t_block_unique.shape[0],2))

    # initialize matrix for p(high_p_port)
    high_p_port = np.zeros_like(p_switch_block)

    '''
    calculate p(switch) for each trial # in block (from 0 -> end)
    '''
    for t in t_block_unique:
        switches = data[data['Block Trial'] == t]['Switch']
        p_switch_block[t,0] = switches.mean(axis=0)
        p_switch_block[t,1] = switches.std(axis=0) / np.sqrt(switches.shape[0])
        
        highport = data[data['Block Trial']==t]['Higher p port']
        high_p_port[t,0] = highport.mean(axis=0)
        high_p_port[t,1] = highport.std(axis=0) / np.sqrt(highport.shape[0])


    '''
    calculate p(switch) and p(high port) for trial #s in block (from -L to +L)
    '''

    data.index = np.arange(data.shape[0]) # <-- this is important
    switch_points = data[data['Block Trial'] == 0].index.values

    L = 30
    paraswitch = np.zeros((switch_points.shape[0],L*2 + 1))
    paraswitch_port = np.zeros_like(paraswitch)

    for i,point in enumerate(switch_points):
        try:
            paraswitch[i,:] = data.iloc[point-L:point+L+1]['Switch']
            paraswitch_port[i,:] = data.iloc[point-L:point+L+1]['Higher p port']
        except:
            pass

    '''
    calculate exponential fit for p(high port) after block switch
    '''
    #first define exponential function to be optimized
    def exp_func(x,a,b,c):
        return a*np.exp(-b*x) + c
    
    #fit curve
    popt,pcov = optimize.curve_fit(exp_func,np.arange(L+1),paraswitch_port.mean(axis=0)[L:])
    
    #calc peak_switch, stable_phigh, and tau
    stable_switch = paraswitch[:,L-10:L].mean()
    peak_switch = paraswitch[:,L:L+10].mean(axis=0).max()
    stable_phigh = paraswitch_port[:,L-10:L].mean()
    rebias_tau = 1./popt[1]
    
    d = {'stable_phigh':stable_phigh,
         'stable_pswitch':stable_switch,
         'peak_pswitch':peak_switch,
         'rebias_tau':rebias_tau}
    
    return pd.DataFrame(data=d,index=[0])