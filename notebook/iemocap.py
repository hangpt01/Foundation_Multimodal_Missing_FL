import os
import pickle
import numpy as np
from typing import Callable, Optional
import torch
import sys
from torch.utils.data import Dataset

def read_data():
    data_file = "/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP/iemocap_train.pkl"
    data,  audios,  texts,  visions,  labels = pickle.load(open(data_file,"rb"), encoding='latin1')
    print(audios.shape,  texts.shape,  visions.shape,  labels.shape)
    # import pdb; pdb.set_trace()
    print(audios[:10, 0],  texts[:10, 0],  visions[:10, 0],  labels[:10])
    
    
if __name__ == '__main__':
    read_data()