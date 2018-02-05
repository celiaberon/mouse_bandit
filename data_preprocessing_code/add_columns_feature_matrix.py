#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:06:05 2018

@author: celia
"""

import os
import pandas as pd
import numpy as np


def add_block_ids(data, record, save=False, user_name='celia'):
    
    
    '''
    This function adds the block columns to a feature matrix:
    
    Inputs:
            data     :  pandas dataframe
            record   :  session record (.csv)
            user     :  username for filepaths 
            save     :  save dataframe 

    Outputs:
            data: pandas dataframe with new column for block_id and current block length
            
    '''
    
    
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame', 'Block ID']
    
    
    
    for session in np.unique(data['Session ID'])[:]:
    
        # load in data from a particular session
        record[record['Session ID'] == session] # take only rows from record that match session name
        root_dir = '/Users/{}/GitHub/mouse_bandit/data/trial_data'.format(user_name)
        full_name = session + '_trials.csv'
        path_name = os.path.join(root_dir,full_name)
        trial_df = pd.read_csv(path_name,names=columns) # load in full dataset from a single trial
        
        # determine block lengths and attach block id to each trial
        # identify blocks by trials where right reward probability changes
        blocks = list(np.diff(np.concatenate((range(1),np.where(np.diff(trial_df['Right Reward Prob']) != 0)[0]+2))))
        
        try: 
            blocks.append(int(record[record['Session ID']==session]['No. Trials'].values)
                     -np.where(np.diff(trial_df['Right Reward Prob']) != 0)[0][-1]-1) # length of the last block at end
            block_id = np.asarray([x+1 for x in range(int(record[record['Session ID']==session]['No. Blocks'].values[0])+1)
                for y in range(blocks[x])])
    
            trial_df['Block ID'] = block_id[:-1] # add column in trial_df for block id
            
            # add block id and current block length to dataframe; start with 11th frame so can fill in full feature matrix
            data.loc[data[data['Session ID']==session].index, 'Block ID'] = block_id[11:]
            temp_blocks = [blocks[x-1] for x in block_id[11:]]
            data.loc[data[data['Session ID']==session].index, 'Current Block Length'] = temp_blocks
        
        except IndexError: # report any errors - days when no blocks were completed
            print('error')
            
    return data
    #if len(error) == True:
    #    return error
        
    
    if save == True:
        save_dir = '/Users/{}/GitHub/mouse_bandit/data/processed_data'.format(user_name)
        data.to_csv(os.path.join(save_dir,'markov_master_labeled.csv'),index=True)