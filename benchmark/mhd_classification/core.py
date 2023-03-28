import os
import ujson
import importlib
import random
from torchvision import transforms
from .dataset import MHDDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import MultimodalClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe

def save_task(generator):
    """
    Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
    The input 'generator' must have attributes:
        :taskpath: string. the path of storing
        :train_data: the training dataset which is a dict {'x':..., 'y':...}
        :test_data: the testing dataset which is a dict {'x':..., 'y':...}
        :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
        :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
        :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
        :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                For example, for MNIST using this task pipe, the source_dict should be like:
                {'class_path': 'torchvision.datasets',
                    'class_name': 'MNIST',
                    'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                    'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                }
        :return:
    """
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    modality_combinations = [
        ["image"],
        ["sound"],
        ["trajectory"],
        ["image", "sound"],
        ["image", "trajectory"],
        ["sound", "trajectory"],
        ["image", "sound", "trajectory"]
    ]
    for cid in range(len(generator.cnames)):
        feddata[generator.cnames[cid]] = {
            'modalities': random.choice(modality_combinations),
            'dtrain': generator.train_cidxs[cid],
            'dvalid': generator.valid_cidxs[cid]
        }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf)
    return

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed= 0):
        super(TaskGen, self).__init__(benchmark='mhd_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/MHD',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        self.num_classes = 10
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.mhd_classification.dataset',
            'class_name': 'MHDDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'image_transform': 'transforms.Compose([transforms.Normalize((0.0841,), (0.2487,))])',
                'trajectory_transform': 'transforms.Compose([transforms.Normalize((0.5347,), (0.1003,))])',
                'sound_transform': 'transforms.Compose([transforms.Normalize((0.4860,), (0.1292,))])',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'image_transform': 'transforms.Compose([transforms.Normalize((0.0841,), (0.2487,))])',
                'trajectory_transform': 'transforms.Compose([transforms.Normalize((0.5347,), (0.1003,))])',
                'sound_transform': 'transforms.Compose([transforms.Normalize((0.4860,), (0.1292,))])',
                'train': 'False'
            }
        }

    def load_data(self):
        self.train_data = MHDDataset(
            root=self.rawdata_path,
            train=True,
            download=True,
            image_transform=transforms.Compose([transforms.Normalize((0.0841,), (0.2487,))]),
            trajectory_transform=transforms.Compose([transforms.Normalize((0.5347,), (0.1003,))]),
            sound_transform=transforms.Compose([transforms.Normalize((0.4860,), (0.1292,))]),
        )
        self.test_data = MHDDataset(
            root=self.rawdata_path,
            train=False,
            download=True,
            image_transform=transforms.Compose([transforms.Normalize((0.0841,), (0.2487,))]),
            trajectory_transform=transforms.Compose([transforms.Normalize((0.5347,), (0.1003,))]),
            sound_transform=transforms.Compose([transforms.Normalize((0.4860,), (0.1292,))]),
        )

class TaskPipe(IDXTaskPipe):
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        origin_class = getattr(importlib.import_module(class_path), class_name)
        origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = []
        valid_datas = []
        modalities_list = []
        for name in feddata['client_names']:
            train_data = feddata[name]['dtrain']
            valid_data = feddata[name]['dvalid']
            if cls._cross_validation:
                k = len(train_data)
                train_data.extend(valid_data)
                random.shuffle(train_data)
                all_data = train_data
                train_data = all_data[:k]
                valid_data = all_data[k:]
            if cls._train_on_all:
                train_data.extend(valid_data)
            train_datas.append(cls.TaskDataset(origin_train_data, train_data))
            valid_datas.append(cls.TaskDataset(origin_train_data, valid_data))
            modalities_list.append(feddata[name]['modalities'])
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list