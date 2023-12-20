import os
import pickle
import numpy as np
from typing import Callable, Optional
import torch
import sys
from torch.utils.data import Dataset

def gen_list_modalities (missing_rate=0.5, missing_ratio_2_modal=0.5, num_modalities=2, NUM_USER=20):
    mat_modals = []
    list_modals_tuples = []
    # num_sample_modals = np.zeros(num_modalities)
    for i in range(NUM_USER):
        unimodal_ind = np.random.randint(num_modalities, size=1)
        # print("Uni: ", np.random.randint(num_modalities, size=1))
        modal_list = np.random.binomial(size=num_modalities, n=1, p=1-missing_rate)
        # modal_indexes = np.where(modal_list==1)[0]
        modal_list[unimodal_ind] = 1
        modal_indexes = np.where(modal_list==1)[0]
        # print(modal_indexes)
        list_modals_tuples.append(tuple(modal_indexes))  
        mat_modals.append(modal_list.tolist())
    mat_modals = np.array(mat_modals)
    num_sample_modals = np.sum(mat_modals, axis=0)
    print("Num_sam:", num_sample_modals)
    print(list_modals_tuples)
    return list_modals_tuples      
    
if __name__ == '__main__':
    gen_list_modalities(1, 2, 20)