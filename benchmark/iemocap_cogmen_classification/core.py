from .dataset import IEMOCAPDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
import os
import time
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
            modalities_list.append(feddata[name]['modalities'])
            # modalities_list.append(list(range(12)))
        # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
    for cid in range(len(generator.cnames)):
        # print(cid)
        if generator.specific_training_leads:
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
    # import pdb; pdb.set_trace()
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf, default=convert)
    return

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)        #cluster=class

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = np.random.randint(0, len(dataset), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}       # class:[indices]
    for idx in selected_indices:
        _, label = dataset[idx]
        # import pdb; pdb.set_trace()
        group_id = label2cluster[int(label)]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_indices = [[] for _ in range(n_clients)]        
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster
    
    for cluster_id in range(n_clusters):
            weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
            clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    
    clients_counts = np.cumsum(clients_counts, axis=1)

    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            # clients_indices[client_id] += list(indices)
            clients_indices[client_id] += indices
            # import pdb; pdb.set_trace()

                    
    return clients_indices


def noniid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.y)
    # import pdb; pdb.set_trace()
    clients_indices = by_labels_non_iid_split(generator.train_data, labels.shape[0], generator.num_clients, labels.shape[0], generator.skewness, frac=1, seed=generator.seed)
    # import pdb; pdb.set_trace()
    return clients_indices

def iid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.y)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    return local_datas

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=23, skewness=0.5, local_hld_rate=0.0, seed=0, percentages=None, missing=False, modal_equality=False, modal_missing_case3=False, modal_missing_case4=False):
        super(TaskGen, self).__init__(benchmark='iemocap_cogmen_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/IEMOCAP_COGMEN',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        # if self.dist_id == 0:
        #     # self.partition = default_partition
        #     self.partition = iid_partition
        if self.dist_id == 1:
            self.partition = noniid_partition
        self.num_classes = 4
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.iemocap_cogmen_classification.dataset',
            'class_name': 'IEMOCAPDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'False',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'False',
                'train': 'False'
            }
        }
        self.num_clients = num_clients
        self.missing = missing
        self.modal_equality = modal_equality
        self.missing_rate_0_3 = modal_missing_case3
        self.modal_missing_case4 = modal_missing_case4
        self.local_holdout_rate = 0.1
        # if self.local_holdout_rate > 0:
        #     self.taskname = self.taskname + '_mifl_gblend'
        if self.modal_equality:         # p=1, #modals_sample = [8 7 5]
            self.specific_training_leads = [(0,),
                                            (1,),
                                            (2,),
                                            (0,),
                                            (0,),
                                            (1,),
                                            (1,),
                                            (2,),
                                            (2,),
                                            (1,),
                                            (0,),
                                            (1,),
                                            (0,),
                                            (2,),
                                            (1,),
                                            (0,),
                                            (1,),
                                            (0,),
                                            (2,),
                                            (0,)]
            self.taskname = self.taskname + '_missing_rate_1'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        elif self.missing_rate_0_3:       # p=0.3, #modals_sample = [14  8 11]  
            self.specific_training_leads = [(2,), 
                                            (0, 1, 2), 
                                            (0, 1), 
                                            (0, 2), 
                                            (0, 2), 
                                            (0, 2), 
                                            (0,), 
                                            (0, 1, 2), 
                                            (2,), 
                                            (0, 2), 
                                            (1, 2), 
                                            (1,), 
                                            (0, 2), 
                                            (1,), 
                                            (0,), 
                                            (0,), 
                                            (0, 1), 
                                            (1,), 
                                            (0, 2), 
                                            (0,)]
            self.taskname = self.taskname + '_missing_rate_0.3'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        else:
            # p=0.7, #modals_sample = [13 11  9] 
            self.specific_training_leads = [(0, 2), 
                                            (0,), 
                                            (0, 1), 
                                            (0, 1, 2), 
                                            (0, 2), 
                                            (1,), 
                                            (0, 1), 
                                            (0, 1, 2), 
                                            (0, 2), 
                                            (1,), 
                                            (1,), 
                                            (1,), 
                                            (2,), 
                                            (0, 1, 2), 
                                            (0,), 
                                            (0, 2), 
                                            (0,), 
                                            (2,), 
                                            (0, 1), 
                                            (1,)]
            self.taskname = self.taskname + '_missing_rate_0.7'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        # self.taskname = self.taskname + '_missing'
        

    def load_data(self):
        self.train_data = IEMOCAPDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=False,
            train=True
        )
        self.test_data = IEMOCAPDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=False,
            train=False
        )
        
    # def partition(self):
    #     # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
    #     if self.dist_id == 0:
    #         """IID"""
    #         # d_idxs = np.random.permutation(len(self.train_data))
    #         # local_datas = np.array_split(d_idxs, self.num_clients)
    #         # local_datas = [data_idx.tolist() for data_idx in local_datas]
    #         local_datas = [data_idx.tolist() for data_idx in self.train_data]
        
    #     return local_datas

    
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
        loss = model(tdata[0], tdata[-1], leads)
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
    def evaluate(self, model, dataset, leads, batch_size=64, num_workers=0):
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
        # import pdb; pdb.set_trace()    

        # total_loss = []
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, loss_, outputs = model(batch_data[0], batch_data[-1], leads)
            # import pdb; pdb.set_trace()    
            loss = torch.tensor(loss).to(loss[0].device)
            if batch_id == 0:
                total_loss = loss
            else:    
                total_loss = loss + total_loss
            
        
        loss_eval = [loss / (batch_id + 1) for loss in total_loss]
        # import pdb; pdb.set_trace()
        #     total_loss += loss.item() * len(batch_data[-1])
        #     predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        # labels = np.array(labels)
        # predicts = np.array(predicts)
        # accuracy = accuracy_score(labels, predicts)
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        return loss_eval

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
        result = dict() 
        for test_combi_index in range(len(leads)):
            total_loss = 0.0
            labels = list()
            predicts = list()    
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                loss, outputs = model(batch_data[0], batch_data[-1], leads[test_combi_index])
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
                import pdb; pdb.set_trace()
                
                # breakpoint()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)
            result['acc'+str(test_combi_index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        # import pdb;pdb.set_trace()
        # import pdb; pdb.set_trace()
        return result


    @torch.no_grad()
    def full_modal_server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
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
    def independent_test(self, model, dataset, leads, batch_size=64, num_workers=0):
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
        result = dict() 
        for test_combi_index in range(len(leads)):
            labels = list()
            predicts = list()
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                predict = model.predict(batch_data[0], batch_data[-1], leads[test_combi_index])
                predicts.extend(predict.argmax(dim=1).cpu().tolist())
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            result['acc'+str(test_combi_index+1)] = accuracy
        return result
        
    @torch.no_grad()
    def independent_test_detail(self, model, dataset, leads, batch_size=64, num_workers=0):
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=1, num_workers=num_workers)
        labels = list()
        
        fin_output = []
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            fin_output.append(model.predict_detail(batch_data[0], batch_data[-1], leads))
        
        return fin_output