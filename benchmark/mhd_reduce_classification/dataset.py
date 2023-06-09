import os
import numpy as np
from torch.utils.data import Dataset

def unstack_array(array, axis=0):
    array_list = list()
    for i in range(array.shape[axis]):
        array_list.append(array[i])
    array_unstack = np.concatenate(array_list, axis=0)
    return array_unstack

class MHDReduceDataset(Dataset):
    def __init__(
        self, 
        root: str = './benchmark/RAW_DATA/MHD_REDUCE',
        train: bool = True
    ) -> None:
        self.train = train

        if self.train:
            self.labels = np.load(os.path.join(root, 'train_label.npy'))
            self.images = np.load(os.path.join(root, 'train_image.npy'))
            self.sounds = np.load(os.path.join(root, 'train_sound.npy'))
            self.trajectories = np.load(os.path.join(root, 'train_trajectory.npy'))
        else:
            self.labels = np.load(os.path.join(root, 'test_label.npy'))
            self.images = np.load(os.path.join(root, 'test_image.npy'))
            self.sounds = np.load(os.path.join(root, 'test_sound.npy'))
            self.trajectories = np.load(os.path.join(root, 'test_trajectory.npy'))
        # import pdb; pdb.set_trace()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, trajectory, sound), label
        """
        image = self.images[index]

        audio = np.expand_dims(unstack_array(self.sounds[index]), axis=0)
        audio_perm = np.transpose(audio, (0, 2, 1))

        trajectory = self.trajectories[index]

        label = self.labels[index]

        return {
            "image": image,
            "sound": audio_perm,
            "trajectory": trajectory
        }, label

    def __len__(self):
        return self.labels.shape[0]
    
if __name__ == '__main__':
    dataset = MHDReduceDataset(
        root='./benchmark/RAW_DATA/MHD_REDUCE',
        train=False
    )
    import pdb; pdb.set_trace()
    dataset[0]