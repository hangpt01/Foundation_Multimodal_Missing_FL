from .dataset import PTBXLDataset
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

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
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads, contrastive_weight, temperature, margin):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        loss, _ = model(tdata[0], tdata[-1], leads, contrastive_weight, temperature, margin)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, leads, contrastive_weight, temperature, margin, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        # print("TYPEEEEEEEEEEEE", model.ln_1.weight.data.dtype)
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            loss, outputs = model(batch_data[0], batch_data[-1], leads, contrastive_weight, temperature, margin)
            batch_mean_loss = loss.item()
            predicts.extend(torch.sigmoid(outputs).cpu().tolist())
            labels.extend(batch_data[1].cpu().tolist())
            total_loss += batch_mean_loss * len(batch_data[-1])
        labels = np.array(labels)
        predicts = np.array(predicts)
        auc = roc_auc_score(labels, predicts, average='macro')
        return {
            'loss': total_loss/len(dataset),
            'mean_auc': auc,
        }

    @torch.no_grad()
    def custom_test(self, model, dataset, contrastive_weight, temperature, margin, batch_size=64, num_workers=0):
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
        total_loss = {
            'all': 0.0,
            '2': 0.0
        }
        labels = list()
        predicts = {
            'all': list(),
            '2': list()
        }
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            for leads in ['all', '2']:
                loss, outputs = model(batch_data[0], batch_data[-1], leads, contrastive_weight, temperature, margin)
                batch_mean_loss = loss.item()
                predicts[leads].extend(torch.sigmoid(outputs).cpu().tolist())
                total_loss[leads] += batch_mean_loss * len(batch_data[-1])
            labels.extend(batch_data[1].cpu().tolist())
        labels = np.array(labels)
        auc = dict()
        for leads in ['all', '2']:
            predicts[leads] = np.array(predicts[leads])
            auc[leads] = roc_auc_score(labels, predicts[leads], average='macro')
        return {
            'all_loss': total_loss['all'] / len(dataset),
            'all_mean_auc': auc['all'],
            '2_loss': total_loss['2'] / len(dataset),
            '2_mean_auc': auc['2'],
        }