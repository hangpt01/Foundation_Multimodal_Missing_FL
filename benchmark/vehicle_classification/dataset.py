from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random

class VEHICLEDataset(Dataset):
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
            # import pdb; pdb.set_trace()
            self.x = np.load(os.path.join(self.root, 'x_train.npy'), allow_pickle=True)
            self.y = np.load(os.path.join(self.root, 'y_train.npy'), allow_pickle=True)
            self.y = np.array(self.y.reshape(-1), dtype='int64')
            # import pdb; pdb.set_trace()
        
        else:
            self.x = np.load(os.path.join(self.root, 'x_test.npy'), allow_pickle=True)
            self.y = np.load(os.path.join(self.root, 'y_test.npy'), allow_pickle=True)
            self.y = np.array(self.y.reshape(-1), dtype='int64')
        
        if self.standard_scaler:
            self.ss = pickle.load(open(os.path.join(self.root, 'standard_scaler.pkl'), 'rb'))
            x_tmp = list()
            for x in self.x:
                x_shape = x.shape
                x_tmp.append(self.ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
            self.x = np.array(x_tmp)
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        # start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
        # x = x[start_idx:start_idx + self.crop_length].transpose().astype(np.float32)
        x = np.transpose(x,(1,0)).reshape(2,50).astype(np.float32)
        y = y.astype(np.float32)
        # y = int(y)
        return x, y
    
if __name__ == '__main__':
    dataset = VEHICLEDataset(root='./benchmark/RAW_DATA/VEHICLE', standard_scaler=False, train=True)