import os
import pickle
import numpy as np
from typing import Callable, Optional
import torch
import sys
from torch.utils.data import Dataset

def gen_list_2_modalities (missing_rate=0.5, missing_ratio_2_modal=0.5, NUM_USER=20):
    list_modals_tuples = []
    count = [0] * 3
    for i in range(int(NUM_USER*missing_rate*(1-missing_ratio_2_modal))):
        list_modals_tuples.append(tuple([0]))
        count[0] += 1
    for i in range(int(NUM_USER*missing_rate*missing_ratio_2_modal)):
        list_modals_tuples.append(tuple([1]))
        count[1] += 1
    for i in range(NUM_USER-len(list_modals_tuples)):
        list_modals_tuples.append(tuple([0,1]))
        count[2] += 1

    print("Num_sam:", count)
    print(list_modals_tuples)
    return list_modals_tuples      
    
if __name__ == '__main__':
    gen_list_2_modalities(0.8, 0.25, 20)
    gen_list_2_modalities(0.8, 0.5, 20)
    gen_list_2_modalities(0.8, 0.75, 20)
    