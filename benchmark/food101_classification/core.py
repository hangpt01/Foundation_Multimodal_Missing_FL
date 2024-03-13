from .dataset import Food101Dataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import warnings
import collections
warnings.filterwarnings('ignore')
    
class TaskPipe(IDXTaskPipe):
    # TaskDataset = Food101Subset
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
            train_data = feddata[name]['dtrain']    # sample idx
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
            # modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list

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
        if generator.specific_training_leads:
            # import pdb; pdb.set_trace()
            feddata[generator.cnames[cid]] = {
                'modalities': generator.specific_training_leads[cid],
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        else:
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf)
    return
    
# def iid_partition(generator):
#     print(generator)
#     # import pdb; pdb.set_trace()
#     labels = np.unique(generator.train_data.y)
#     local_datas = [[] for _ in range(generator.num_clients)]
#     for label in labels:
#         permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
#         split = np.array_split(permutation, generator.num_clients)
#         for i, idxs in enumerate(split):
#             local_datas[i] += idxs.tolist()
#     # import pdb; pdb.set_trace()
#     return local_datas


def iid_partition(generator):
    print(generator)
    # import pdb; pdb.set_trace()
    labels = np.unique(generator.train_data.get_all_labels())
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.get_all_labels() == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    # import pdb; pdb.set_trace()
    return local_datas

# def local_holdout(self, local_datas, shuffle=False):
#         """split each local dataset into train data and valid data according the rate."""
#         train_cidxs = []
#         valid_cidxs = []
#         for local_data in local_datas:
#             if shuffle:
#                 np.random.shuffle(local_data)
#             k = int(len(local_data) * (1-self.local_holdout_rate))
#             train_cidxs.append(local_data[:k])
#             valid_cidxs.append(local_data[k:])
#         return train_cidxs, valid_cidxs

def get_food101_dataloader(dataset, batch_size=40, shuffle=True, num_workers=8, vocab_size=30522):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate)
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, missing=False):
        super(TaskGen, self).__init__(benchmark='food101_classification',
                                      dist_id=dist_id, 
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/FOOD101',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed)
        if self.dist_id==0:
            self.partition = iid_partition
        
        self.num_classes=101
        self.save_task=save_task
        self.visualize=self.visualize_by_class_food101
        self.source_dict = {
            'class_path': 'benchmark.food101_classification.dataset',
            'class_name': 'Food101Dataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'False',
                'train': 'True',
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'False',
                'train':' False'
            }
        }
        
        self.missing=missing
        self.local_holdout_rate = 0.1
        self.specific_training_leads = None
        
        if self.missing and self.num_clients==10:
            self.specific_training_leads = [[0, 1]]*10, [[0]]*5 + [[1]]*5 
            self.taskname = self.taskname + '_missing'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        # if self.missing and self.num_clients==20:
        #     self.specific_training_leads = [[0, 1]]*10, [[0]]*5 + [[1]]*5 
        #     self.taskname = self.taskname + '_clip_local_missing'
        #     self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        # if self.missing and self.num_clients==10:
        #     self.specific_training_leads = [[0, 1]]*6 + [[0]]*2 + [[1]]*2
        #     self.taskname = self.taskname + '_clip_local_missing'
        #     self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        # if self.subset and self.num_clients==10:
        #     self.specific_training_leads = [[0, 1]]*10
        #     self.taskname = self.taskname + '_clip_local_full_modal_subset'
        #     self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        # if self.subset and self.num_clients>1 and not self.missing:
        #     self.specific_training_leads = [[0, 1]]*self.num_clients
        #     self.taskname = self.taskname + '_clip_local_full_modal_subset'
        #     self.taskpath = os.path.join(self.task_rootpath, self.taskname)            
        # else:
        #     self.specific_training_leads = [[0, 1]]*self.num_clients
        #     self.taskname = self.taskname + '_clip_local_full_modal'
        #     self.taskpath = os.path.join(self.task_rootpath, self.taskname)
            
            
    def load_data(self):
        self.train_data = Food101Dataset(
            root=self.rawdata_path,
            download=False,
            train=True
        )
        # import pdb; pdb.set_trace()
        self.test_data = Food101Dataset(
            root=self.rawdata_path,
            download=False,
            train=False
        )
        
        
    def local_holdout(self, local_datas, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * (1-self.local_holdout_rate))
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        
        return train_cidxs, valid_cidxs
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads=2
        self.DataLoader = DataLoader
        
    def get_optimizer(self, model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        OPTIM = getattr(importlib.import_module('torch.optim'),  self.optimizer_name)
        # filter_fn = filter(lambda p: p.requires_grad, model.parameters())
        clip_params = [p for n, p in model.named_parameters() if (p.requires_grad and ('clip' in n))]
        base_params = [p for n, p in model.named_parameters() if (p.requires_grad and ('clip' not in n))]

        print("Number of CLIP parameters:", len(clip_params))
        print("Number of additional parameters:", len(base_params))
        if self.optimizer_name.lower() == 'sgd':
            return OPTIM(base_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer_name.lower() in ['adam', 'rmsprop', 'adagrad']:
            return OPTIM(base_params, lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError("Invalid Optimizer.")

    def data_to_device(self, data):
        new_data = [x.to(self.device) for x in data]
        return new_data
        
    def train_one_step(self, model, data, leads):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        # import pdb; pdb.set_trace()
        model.to(self.device) # y.device
        # print(tdata[0])
        loss, _ = model([tdata[0], tdata[1]], tdata[-1], leads)
        # print(loss.cpu().item())
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
        data_loader = get_food101_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[-1].cpu().tolist())
            # import pdb; pdb.set_trace()
            loss, outputs = model([batch_data[0], batch_data[1]], batch_data[-1], leads)
            total_loss += loss.item()
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / (batch_id+1),
            'acc': accuracy
        }
    
    @torch.no_grad()
    def server_test(self, model, prompt_pools, dataset, leads, batch_size=1, num_workers=0, ada=False):
        """
        Test metric on global model
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param prompt_pools:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        model.to(self.device)
        assert batch_size==1, 'Only infer a single sample'
        data_loader = get_food101_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        mode = "Adaptive forward" if ada else "Normal forward"
        print("Current forward mode: ", mode)
        for batch_id, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_data = self.data_to_device(batch_data)
            if ada==1:
                loss, output = model.adaptive_forward([batch_data[0], batch_data[1]], batch_data[-1], prompt_pools=prompt_pools)
            else:
                loss, output = model([batch_data[0], batch_data[1]], batch_data[-1], leads)
            total_loss += loss.item() # batch size = 1
            predicts.extend(torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().tolist())
            labels.extend(batch_data[-1].cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        
        print("Test accuracy:", accuracy)
        return {
            'loss': total_loss / (batch_id + 1),
            'acc': accuracy
        }
        
    @torch.no_grad()
    def evaluate(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Evaluate metric on client model
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1: batch_size=len(dataset)
        data_loader = get_food101_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            loss, outputs = model(batch_data[0], batch_data[1], leads)
            if batch_id==0:
                total_loss = loss
            else:
                total_loss = loss + total_loss
        loss_eval = loss / (batch_id + 1) 
        return loss_eval
        