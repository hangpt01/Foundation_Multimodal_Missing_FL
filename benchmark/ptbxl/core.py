from .dataset import PTBXLDataset
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
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
            modalities_list.append(feddata[name]['modalities'])
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads, contrastive_weight, temperature, margin, kl_weight):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        loss_allleads, outputs_allleads, loss_2leads, outputs_2leads = model(tdata[0], tdata[-1], leads, contrastive_weight, temperature, margin, kl_weight)
        if leads == 'all':
            return {'loss': loss_allleads + loss_2leads}
        else:
            return {'loss': loss_2leads}

    @torch.no_grad()
    def test(self, model, dataset, leads, contrastive_weight, temperature, margin, kl_weight, batch_size=64, num_workers=0):
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
            labels.extend(batch_data[1].cpu().tolist())
            loss_allleads, outputs_allleads, loss_2leads, outputs_2leads = model(batch_data[0], batch_data[-1], leads, contrastive_weight, temperature, margin, kl_weight)
            total_loss['2'] += loss_2leads.item() * len(batch_data[-1])
            predicts['2'].extend(torch.sigmoid(outputs_2leads).cpu().tolist())
            if np.any(np.isnan(torch.sigmoid(outputs_2leads).cpu().numpy())):
                import pdb; pdb.set_trace()
            if leads == 'all':
                total_loss['all'] += loss_allleads.item() * len(batch_data[-1])
                predicts['all'].extend(torch.sigmoid(outputs_allleads).cpu().tolist())
                if np.any(np.isnan(torch.sigmoid(outputs_allleads).cpu().numpy())):
                    import pdb; pdb.set_trace()
        labels = np.array(labels)
        auprc = dict()
        for leads_ in ['all', '2']:
            predicts[leads_] = np.array(predicts[leads_])
            auprc[leads_] = average_precision_score(labels, predicts[leads_], average='weighted')
        if leads == 'all':
            return {
                'all_loss': total_loss['all'] / len(dataset),
                'all_mean_auprc': auprc['all'],
                '2_loss': total_loss['2'] / len(dataset),
                '2_mean_auprc': auprc['2'],
            }
        else:
            return {
                '2_loss': total_loss['2'] / len(dataset),
                '2_mean_auprc': auprc['2'],
            }

    @torch.no_grad()
    def server_test(self, model, dataset, contrastive_weight, temperature, margin, kl_weight, batch_size=64, num_workers=0):
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
            labels.extend(batch_data[1].cpu().tolist())
            loss_allleads, outputs_allleads, loss_2leads, outputs_2leads = model(batch_data[0], batch_data[-1], 'all', contrastive_weight, temperature, margin, kl_weight)
            total_loss['2'] += loss_2leads.item() * len(batch_data[-1])
            predicts['2'].extend(torch.sigmoid(outputs_2leads).cpu().tolist())
            if np.any(np.isnan(torch.sigmoid(outputs_2leads).cpu().numpy())):
                import pdb; pdb.set_trace()
            total_loss['all'] += loss_allleads.item() * len(batch_data[-1])
            predicts['all'].extend(torch.sigmoid(outputs_allleads).cpu().tolist())
            if np.any(np.isnan(torch.sigmoid(outputs_allleads).cpu().numpy())):
                import pdb; pdb.set_trace()
        labels = np.array(labels)
        auprc = dict()
        for leads in ['all', '2']:
            predicts[leads] = np.array(predicts[leads])
            auprc[leads] = average_precision_score(labels, predicts[leads], average='weighted')
        return {
            'all_loss': total_loss['all'] / len(dataset),
            'all_mean_auprc': auprc['all'],
            '2_loss': total_loss['2'] / len(dataset),
            '2_mean_auprc': auprc['2'],
        }