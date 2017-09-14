#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:04:38 2017

@author: celia
"""

import sys
sys.path.append('/Users/celia/GitHub/mouse_bandit/helper_functions')
sys.path.append('/Users/celia/GitHub/mouse_bandit')
import numpy as np
import sys

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