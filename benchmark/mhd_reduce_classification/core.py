import os
import ujson
import importlib
import random
from .dataset import MHDReduceDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import torch
import numpy as np
from itertools import combinations
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import random

def save_task(generator):
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    for cid in range(len(generator.cnames)):
        feddata[generator.cnames[cid]] = {
            'modalities': generator.all_cases[cid],
            'dtrain': generator.train_cidxs[cid],
            'dvalid': generator.valid_cidxs[cid]
        }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf)
    return

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.0, seed= 0, missing=False):
        super(TaskGen, self).__init__(benchmark='mhd_reduce_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/MHD_REDUCE',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        self.num_classes = 10
        if missing:
            self.all_cases = random.choices(
                population=[
                    ['image'],
                    ['sound'],
                    ['trajectory'],
                    ['image', 'sound'],
                    ['image', 'trajectory']
                ],
                k=self.num_clients
            )
            self.taskname = self.rename_task()
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        else:
            self.all_cases = [['image', 'sound', 'trajectory']] * self.num_clients
        # import pdb; pdb.set_trace()
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.mhd_reduce_classification.dataset',
            'class_name': 'MHDReduceDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'train': 'False'
            }
        }

    def rename_task(self):
        """Create task name and return it."""
        taskname = '_'.join([
            self.benchmark,
            'cnum' +  str(self.num_clients),
            'dist' + str(self.dist_id),
            'skew' + str(self.skewness).replace(" ", ""),
            'seed'+str(self.seed),
            'missing'
        ])
        return taskname

    def load_data(self):
        self.train_data = MHDReduceDataset(
            root=self.rawdata_path,
            train=True,
        )
        self.test_data = MHDReduceDataset(
            root=self.rawdata_path,
            train=False,
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