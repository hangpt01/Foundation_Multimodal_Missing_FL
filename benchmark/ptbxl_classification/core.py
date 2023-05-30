from .dataset import PTBXLReduceDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
            # modalities_list.append(feddata[name]['modalities'])
            modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list
    
class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.0, seed= 0, percentages=None):
        super(TaskGen, self).__init__(benchmark='ptbxl_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PTBXL_REDUCE',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        self.num_classes = 10
        self.save_task = TaskPipe.save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.ptbxl_classification.dataset',
            'class_name': 'PTBXLReduceDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'standard_scaler': 'True',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'standard_scaler': 'True',
                'train': 'False'
            }
        }
    
    def load_data(self):
        self.train_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            standard_scaler=True,
            train=True
        )
        self.test_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            standard_scaler=True,
            train=False
        )
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        loss, outputs = model(tdata[0], tdata[-1], leads)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }

    @torch.no_grad()
    def server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }