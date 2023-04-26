import os
from typing import Callable, Optional
import torch
import sys
sys.path.append("benchmark/mosei_classification/")
from torch.utils.data import Dataset


class MOSEIDataset(Dataset):
    def __init__(
        self, 
        root: str = './benchmark/RAW_DATA/MOSEI',
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
            self.data_file = os.path.join(root, "mosei_train_a.dt")
        else:
            self.data_file = os.path.join(root, "mosei_test_a.dt")

        if not os.path.exists(self.data_file):
            if download:
                print('Downloading MOSEI Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/mosei_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        # Load data
        # data_mosei = torch.load(self.data_file)
        # (_, self.texts, self.audios, self.visions), self.labels = data_mosei[:][0], data_mosei[:][1]
        data_mosei = torch.load(self.data_file)
        (_, self.texts, __, self.visions), self.labels = data_mosei[:][0], data_mosei[:][1]

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

        # audio = self.audios[index]
        # if self.audio_transform is not None:
        #     audio = self.audio_transform(audio)

        vision = self.visions[index]
        if self.vision_transform is not None:
            vision = self.vision_transform(vision)
            
        label = self.labels[index].item()
        if self.target_transform is not None:
            label = self.target_transform(label)

        # return (text, audio, audio), label
        return {
            "text": text,       # (50, 300)
            # "audio": audio,     # (50, 74)
            "vision": vision    # (50, 35)
        }, label


    def __len__(self):
        return self.labels.shape[0]
    
if __name__ == '__main__':
    dataset = MOSEIDataset(
        root='./benchmark/RAW_DATA/MOSEI',
        train=False,
        download=True
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=5)
    # a = next(iter(loader))
    # print(a)
    import time
    for batch in loader:
        time.sleep(50)
    # import pdb; pdb.set_trace()