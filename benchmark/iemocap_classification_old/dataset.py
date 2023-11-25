import os
import pickle
import numpy as np
from typing import Callable, Optional
import torch
import sys
sys.path.append("benchmark/iemocap_classification/")
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):
    def __init__(
        self, 
        root: str = './benchmark/RAW_DATA/IEMOCAP_COGMEN',
        train: bool = True,
        text_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        vision_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.train = train
        self.text_transform = text_transform
        self.audio_transform = audio_transform
        self.vision_transform = vision_transform
        self.target_transform = target_transform

        if train:
            self.data_file = os.path.join(root, "iemocap_train.pkl")
        else:
            self.data_file = os.path.join(root, "iemocap_test.pkl")

        if not os.path.exists(self.data_file):
            if download:
                print('Downloading IEMOCAP Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/iemocap_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        # (_, self.texts, __, self.visions), self.labels
        self.data, self.audios, self.texts, self.visions, self.labels = pickle.load(open(self.data_file,"rb"), encoding='latin1')
        # print(self.data.shape, self.audios.shape, self.texts.shape, self.visions.shape, self.labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (text, audio, vision), label
        """ 
        text = self.texts[index]        
        if self.text_transform is not None:
            text = self.text_transform(text)

        audio = self.audios[index]
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        vision = self.visions[index]
        if self.vision_transform is not None:
            vision = self.vision_transform(vision)
            
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        # return (text, audio, audio), label
        return {
            "text": np.reshape(text,(1,-1)),       # (10, 100)
            "audio": np.reshape(audio,(1,-1)),     # (10, 100)
            "vision": np.reshape(vision,(1,-1))    # (10, 512)
        }, label


    def __len__(self):
        return self.labels.shape[0]
    
if __name__ == '__main__':
    dataset = IEMOCAPDataset(
        root='./benchmark/RAW_DATA/IEMOCAP',
        train=False,
        download=False
    )
    temp, label = dataset[0]
    print(temp['text'].shape, temp['audio'].shape, temp['vision'].shape)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=5)
    a = next(iter(loader))
    print(a[0]['text'].shape, a[0]['audio'].shape, a[0]['vision'].shape, a[1].shape)
    # import time
    # for batch in loader:
    #     time.sleep(50)
    # import pdb; pdb.set_trace()