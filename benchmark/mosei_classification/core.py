import os
import ujson
import importlib
import random
from torchvision import transforms
from .dataset import MOSEIDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import torch
import numpy as np
from itertools import combinations
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score

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
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed= 0, percentages=None):
        super(TaskGen, self).__init__(benchmark='mosei_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/MOSEI',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        self.modalities = ["text", "vision"]
        self.num_classes = 10
        if len(percentages) != 2 or sum(percentages) > 1.0:
            self.percentages = [0.0, 0.0]
        else:
            self.percentages = percentages
        self.combin_cnt = [int(p * self.num_clients) for p in self.percentages]
        self.combin_cnt.append(self.num_clients - sum(self.combin_cnt))
        self.taskname = self.rename_task()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        print("Count for each combinations:", self.percentages, self.combin_cnt)
        self.all_cases = [[self.modalities[0]]] * self.combin_cnt[0] + \
                         [[self.modalities[1]]] * self.combin_cnt[1] + \
                         [self.modalities] * self.combin_cnt[2]
        random.shuffle(self.all_cases)
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        
        self.source_dict = {
            'class_path': 'benchmark.mosei_classification.dataset',
            'class_name': 'MOSEIDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
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
            '+'.join(self.modalities),
            '+'.join([str(c) for c in self.combin_cnt])
        ])
        return taskname

    def load_data(self):
        self.train_data = MOSEIDataset(
            root=self.rawdata_path,
            train=True,
            download=True,
        )
        self.test_data = MOSEIDataset(
            root=self.rawdata_path,
            train=False,
            download=True,
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
    

class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)

    def data_to_device(self, data):
        sample_to_device = dict()
        for modal, modal_data in data[0].items():
            sample_to_device[modal] = modal_data.to(self.device)
        return sample_to_device, data[1].to(self.device)
    
    def train_one_step(self, model, data, contrastive_weight, temperature):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        loss, _ = model(tdata[0], tdata[1], contrastive_weight, temperature)
        return {'loss': loss}
    
    @torch.no_grad()
    def test(self, model, dataset, contrastive_weight, temperature, batch_size=64, num_workers=0):
        model.eval()
        if batch_size == -1:
            batch_size = len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = 0.0
        total_count = 0
        labels = list()
        preds = list()
        for data in data_loader:        
            tdata = self.data_to_device(data)
            total_count += tdata[1].shape[0]
            labels.extend(tdata[1].cpu().tolist())
            loss, outputs = model(tdata[0], tdata[1], contrastive_weight, temperature)
            total_loss += loss * tdata[1].shape[0]
            preds.extend(outputs.cpu().tolist())
        truth = np.array(labels)
        truth_a7 = np.clip(truth, a_min=-3.0, a_max=3.0)
        preds = np.array(preds)
        preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
        binary_truth = (truth > 0)
        binary_preds = (preds > 0)
        return {
            'loss': total_loss.item() / total_count,
            'mae': np.mean(np.absolute(preds - truth)),
            'corr': np.corrcoef(preds, truth)[0, 1],
            'acc7': np.sum(np.round(preds_a7) == np.round(truth_a7)) / total_count,
            'acc2': accuracy_score(binary_truth, binary_preds),
            'f1': f1_score(binary_truth, binary_preds, average='weighted'),
        }

    @torch.no_grad()
    def custom_test(self, model, dataset, contrastive_weight, temperature, batch_size=64, num_workers=0, all_modal_combin_list=[]):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size == -1:
            batch_size = len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_count = 0
        labels = list()
        eval_dict = dict()
        for combin_list in all_modal_combin_list:
            combin = "+".join(combin_list)
            eval_dict[combin + "_total_loss"] = 0.0
            eval_dict[combin + "_outputs"] = list()
        for data in data_loader:        
            tdata = self.data_to_device(data)
            total_count += tdata[1].shape[0]
            labels.extend(tdata[1].cpu().tolist())
            for combin_list in all_modal_combin_list:
                combin = "+".join(combin_list)
                samples = dict()
                for modal in combin_list:
                    samples[modal] = tdata[0][modal].contiguous()
                loss, outputs = model(samples, tdata[1], contrastive_weight, temperature)
                eval_dict[combin + "_total_loss"] += loss.item() * tdata[1].shape[0]
                eval_dict[combin + "_outputs"].extend(outputs.cpu().tolist())
        result = dict()
        truth = np.array(labels)
        truth_a7 = np.clip(truth, a_min=-3.0, a_max=3.0)
        for combin_list in all_modal_combin_list:
            combin = "+".join(combin_list)
            result[combin + "_loss"] = eval_dict[combin + "_total_loss"] / total_count
            preds = np.array(eval_dict[combin + "_outputs"])
            preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
            result[combin + "_mae"] = np.mean(np.absolute(preds - truth))
            result[combin + "_corr"] = np.corrcoef(preds, truth)[0, 1]
            result[combin + "_acc7"] = np.sum(np.round(preds_a7) == np.round(truth_a7)) / total_count
            binary_truth = (truth > 0)
            binary_preds = (preds > 0)
            result[combin + "_acc2"] = accuracy_score(binary_truth, binary_preds)
            result[combin + "_f1"] = f1_score(binary_truth, binary_preds, average='weighted')
        return result