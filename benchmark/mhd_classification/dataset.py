import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torchvision import transforms
from torch.utils.data import Dataset

def unstack_tensor(tensor, dim=0):
    tensor_list = list()
    for i in range(tensor.size(dim)):
        tensor_list.append(tensor[i])
    tensor_unstack = torch.cat(tensor_list, dim=0)
    return tensor_unstack

class MHDDataset(Dataset):
    def __init__(
        self, 
        root: str = './benchmark/RAW_DATA/MHD',
        train: bool = True,
        image_transform: Optional[Callable] = None,
        trajectory_transform: Optional[Callable] = None,
        sound_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.train = train
        self.image_transform = image_transform
        self.trajectory_transform = trajectory_transform
        self.sound_transform = sound_transform
        self.target_transform = target_transform

        if train:
            self.data_file = os.path.join(root, "mhd_train.pt")
        else:
            self.data_file = os.path.join(root, "mhd_test.pt")

        if not os.path.exists(self.data_file):
            if download:
                print('Downloading MHD Dataset...', end=' ')
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/mhd_classification/download.sh')
                print('done!')
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        # Load data
        self.labels, \
            self.images, \
            self.trajectories, \
            self.sounds, \
            _, \
            __ = torch.load(self.data_file)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, trajectory, sound), label
        """
        image = self.images[index]
        if self.image_transform is not None:
            image = self.image_transform(image)
        # import pdb; pdb.set_trace()

        audio = unstack_tensor(self.sounds[index]).unsqueeze(0)
        audio_perm = audio.permute(0, 2, 1)
        if self.sound_transform is not None:
            audio_perm = self.sound_transform(audio_perm)

        trajectory = self.trajectories[index]
        if self.trajectory_transform is not None:
            trajectory = self.trajectory_transform(trajectory.view(1, 1, -1)).squeeze()

        label = int(self.labels[index])
        if self.target_transform is not None:
            label = self.target_transform(label)

        # return (image, trajectory, audio_perm), label
        return {
            "image": image,
            "sound": audio_perm,
            "trajectory": trajectory
        }, label


    def __len__(self):
        return self.labels.shape[0]
    
if __name__ == '__main__':
    dataset = MHDDataset(
        root='./benchmark/RAW_DATA/MHD',
        train=False,
        download=True
    )
    dataset[0]
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=5)
    # for batch in loader:
    #     import pdb; pdb.set_trace()
    #     break