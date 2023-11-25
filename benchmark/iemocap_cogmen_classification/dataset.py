from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch
import pickle
import random

class IEMOCAPDataset(Dataset):
    def __init__(self, root, download=True, standard_scaler=False, train=True, crop_length=0):
        self.root = root
        self.standard_scaler = standard_scaler
        self.train = train
        self.crop_length = crop_length
        # import pdb; pdb.set_trace()
        if not os.path.exists(self.root):
            if download:
                print('Downloading IEMOCAP Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/iemocap_cogmen_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        video_ids, video_speakers, video_labels, video_text, \
            video_audio, video_visual, video_sentence, trainVids, \
            test_vids = pickle.load(open(os.path.join(self.root, 'IEMOCAP_features.pkl'),"rb"), encoding='latin1')
        
        if self.train:
            # import pdb; pdb.set_trace()
            self.x = torch.load(os.path.join(self.root, 'x_train.pt'))
            self.y = torch.load(os.path.join(self.root, 'y_train.pt'))
            
            # import pdb; pdb.set_trace()
        
        else:
            self.x = torch.load(os.path.join(self.root, 'x_test.pt'))
            self.y = torch.load(os.path.join(self.root, 'y_test.pt'))

    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        # start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
        # x = x[start_idx:start_idx + self.crop_length].transpose().astype(np.float32)
        # x = np.transpose(x,(1,0)).reshape(2,50).astype(np.float32)
        # y = y.astype(np.float32)
        # y = int(y)
        return x, y
    
if __name__ == '__main__':
    dataset = IEMOCAPDataset(root='./benchmark/RAW_DATA/IEMOCAP_COGMEN', standard_scaler=False, train=True)