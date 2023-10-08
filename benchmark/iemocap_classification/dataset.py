import os
import pickle
import numpy as np
from typing import Callable, Optional
import torch
import sys
sys.path.append("benchmark/iemocap_classification/")
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    def __init__(self, root, download=True, standard_scaler=False, train=True, crop_length=0):
        self.root = root
        self.standard_scaler = standard_scaler
        self.train = train
        self.crop_length = crop_length
        # import pdb; pdb.set_trace()
        if not os.path.exists(self.root):
            if download:
                print('Downloading VEHICLE Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/vehicle_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        if self.train:
            self.data_file = os.path.join(root, "iemocap_train.pkl")
        else:
            self.data_file = os.path.join(root, "iemocap_test.pkl")

        # shapes - audio, text, vision: (5810, 100) (5810, 100) (5810, 512)
        data,  audios,  texts,  visions,  labels = pickle.load(open(self.data_file,"rb"), encoding='latin1')
        self.x = data
        self.y = labels.reshape(-1)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        # start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
        # x = x[start_idx:start_idx + self.crop_length].transpose().astype(np.float32)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # y = int(y)
        return x, y
    
# if __name__ == '__main__':
#     dataset = IEMOCAPDataset(
#         root='./benchmark/RAW_DATA/IEMOCAP',
#         train=False,
#         download=False
#     )
#     temp, label = dataset[0]
#     print(temp['text'].shape, temp['audio'].shape, temp['vision'].shape)
#     from torch.utils.data import DataLoader
#     loader = DataLoader(dataset, batch_size=5)
#     a = next(iter(loader))
#     print(a[0]['text'].shape, a[0]['audio'].shape, a[0]['vision'].shape, a[1].shape)
#     # import time
    # for batch in loader:
    #     time.sleep(50)
    # import pdb; pdb.set_trace()