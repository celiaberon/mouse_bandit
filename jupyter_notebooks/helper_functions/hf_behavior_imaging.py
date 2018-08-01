#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:04:38 2017

@author: celia
"""

import numpy as np
#import sys
#sys.path.append('/Users/{}/GitHub/mouse_bandit/jupyter_notebooks/helper_functions'.format(user_name))
#import hf_behavior_imaging as hf
import numpy as np
import itertools

def extract_frames(df, cond1_name, cond1=False, cond2_name=False,cond2=False, cond3_name=False,
                   cond3=False, cond1_ops= '=', cond2_ops = '=', cond3_ops = '='):
                    # First define function so it can take multiple conditions
    import operator
    
    # set up operator dictionary
    ops = {'>': operator.gt,
       '<': operator.lt,
       '>=': operator.ge,
       '<=': operator.le,
       '=': operator.eq}
    
    if type(cond3_name)==str:
        frames_c = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['center_frame'])
        frames_d = (df[((ops[cond1_ops](df[cond1_name],cond1)) 
                    & (ops[cond2_ops](df[cond2_name], cond2))
                    & (ops[cond3_ops](df[cond3_name],cond3)))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    elif type(cond2_name)==str:
        frames_c = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['center_frame'])
        frames_d = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    else:
        frames_c =(df[(df[cond1_name] == cond1)]['center_frame'])
        frames_d =(df[(df[cond1_name] == cond1)]['decision_frame'])
        frames = np.column_stack((frames_c, frames_d))
        return frames
    
    
def align_frames(df, ca_data, event, cond1_name, cond1=False, cond2_name=False,cond2=False, cond3_name=False,
                   cond3=False, cond1_ops= '=', cond2_ops = '=', cond3_ops = '=', extension=30):
    
    conditions = [cond1_name, cond2_name, cond3_name]
    n_variables = 3 - conditions.count(0) # value between 1 and 3 to run the corresponding number of conditions through the rest of the notebook
    combo_id = list(itertools.product([0, 1], repeat=n_variables)) # list of all combinations of binary conditions
    n_combos = len(combo_id) # total number of combinations of conditions
    
    poke_frames = {} # initialize an empty dictionary to store center and decision poke frames for each combination of conditions
    for i in range(0,n_combos):
        if n_variables==3: # based on number of variables specified
            iCombo_key= "f%s%s%s" % (combo_id[i][0], combo_id[i][1], combo_id[i][2]) # set key name based on comboID
            iCombo_value = extract_frames(df, cond1_name, combo_id[i][0], cond2_name, 
                                             combo_id[i][1], cond3_name, combo_id[i][2], cond1_ops=cond1_ops)
            temp_dict = {iCombo_key:iCombo_value} # create new key-value pair with comboID and corresponding frames
            poke_frames.update(temp_dict) # add new values to dictionary of center and decision poke frames
        if n_variables==2:
            iCombo_key= "f%s%s" % (combo_id[i][0], combo_id[i][1])
            iCombo_value = extract_frames(df, cond1_name, combo_id[i][0], cond2_name, 
                                             combo_id[i][1], cond1_ops=cond1_ops)
            temp_dict = {iCombo_key:iCombo_value}
            poke_frames.update(temp_dict)
        if n_variables==1:
            iCombo_key= "f%s" %(combo_id[i][0])
            iCombo_value = extract_frames(df, cond1_name, combo_id[i][0], cond1_ops=cond1_ops)
            temp_dict = {iCombo_key:iCombo_value}
            poke_frames.update(temp_dict)

    poke_frames_keys = list(poke_frames.keys())
    
    for i in poke_frames: # create full window based on number of frames in 'extension' variable
        poke_frames[i][:,0] = poke_frames[i][:,0] - extension
        poke_frames[i][:,1] = poke_frames[i][:,1] + extension
    
    nTrials = [poke_frames[poke_frames_keys[i]].shape[0] for i in range(n_combos)]
    max_window = np.zeros(n_combos) 
    window_length= np.zeros((np.max(nTrials), n_combos))

    for i in range(n_combos):
        for iTrial in range(nTrials[i]):
            window_length[iTrial, i] = int(((poke_frames[poke_frames_keys[i]][iTrial][1]-
                                     poke_frames[poke_frames_keys[i]][iTrial][0])))
        max_window[i] = np.max(window_length)

    max_window = int(max_window.max())
    
    nNeurons = ca_data.shape[0]
    nFrames = ca_data.shape[1]
    
    aligned_traces = np.empty((np.max(nTrials), max_window, nNeurons, n_combos))

    if event == 'centerPoke':
        for i in range(n_combos):
            
            # create array containing segment of raw trace for each neuron for each trial aligned to center poke
            for iNeuron in range(nNeurons): # for each neuron
                for iTrial in range(0,nTrials[i]): # and for each trial
                    try:
                        aligned_traces[iTrial,:, iNeuron, i] = ca_data[iNeuron,
                            int(poke_frames[poke_frames_keys[i]][iTrial][0]):
                            (int(poke_frames[poke_frames_keys[i]][iTrial][0])+max_window)]
                    except ValueError: # for trials at end of session where cannot fill full max_window length
                        len_to_end = int(nFrames - poke_frames[poke_frames_keys[i]][iTrial][0])
                        aligned_traces[iTrial,0:len_to_end, iNeuron, i] = ca_data[iNeuron,
                            int(poke_frames[poke_frames_keys[i]][iTrial][0]):
                            (int(poke_frames[poke_frames_keys[i]][iTrial][0])+len_to_end)]

    elif event == 'decisionPoke':
        for i in range(n_combos):

            # create array containing segment of raw trace for each neuron for each trial aligned to decision poke
            for iNeuron in range(nNeurons): # for each neuron
                for iTrial in range(0,nTrials[i]): # and for each trial
                    try:
                        aligned_traces[iTrial,:, iNeuron, i] = ca_data[iNeuron,
                            int(poke_frames[poke_frames_keys[i]][iTrial][0]):
                            (int(poke_frames[poke_frames_keys[i]][iTrial][0])+max_window)]
                    except ValueError: # for trials at end of session where cannot fill full max_window length
                        len_to_end = int(nFrames - poke_frames[poke_frames_keys[i]][iTrial][0])
                        #aligned_start[iTrial,0:len_to_end, iNeuron, i] = normalized[iNeuron,
                        #    int(poke_frames[poke_frames_keys[i]][iTrial][0]):
                        #    (int(poke_frames[poke_frames_keys[i]][iTrial][0])+len_to_end)]
                        aligned_traces[iTrial,0:len_to_end, iNeuron, i] = 'NaN'

    return poke_frames_keys, aligned_traces, nTrials