from .dataset import PTBXL_REDUCE_Dataset
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
    # import pdb; pdb.set_trace()
    labels = np.unique(generator.train_data.y)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
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
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, missing_1_6=False, missing_all_6=False, missing_1_12=False, missing_7_12=False, missing_rate=-1, missing_ratio_2_modal=-1):
        super(TaskGen, self).__init__(benchmark='ptbxl_reduce_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PTBXL_REDUCE',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        if self.dist_id == 1:
            self.partition = noniid_partition
        else:
            self.partition = iid_partition
        # self.local_holdout = local_holdout
        self.num_classes = 5
        # self.num_classes = 4
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'benchmark.ptbxl_reduce_classification.dataset',
            'class_name': 'PTBXL_REDUCE_Dataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train':'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train': 'False'
            }
        }
        self.missing_1_6 = missing_1_6
        self.missing_all_6 = missing_all_6
        self.missing_1_12 = missing_1_12
        self.missing_7_12 = missing_7_12
        self.specific_training_leads = None
        self.local_holdout_rate = 0.1
        if self.missing_all_6:
            self.specific_training_leads = [(3, 6, 2, 7, 5, 1), (11, 2, 10, 9, 6, 1), (10, 0, 9, 6, 3, 5), (2, 1, 9, 5, 7, 6), (7, 11, 10, 8, 4, 5), 
                                            (8, 9, 11, 7, 2, 10), (4, 11, 3, 0, 6, 2), (2, 0, 5, 9, 10, 11), (11, 6, 3, 9, 8, 2), (10, 9, 5, 7, 4, 11), 
                                            (4, 1, 7, 2, 11, 10), (10, 1, 2, 4, 8, 9), (2, 4, 7, 0, 5, 10), (9, 10, 5, 6, 7, 1), (1, 4, 10, 8, 6, 11), 
                                            (2, 9, 0, 6, 10, 1), (11, 9, 2, 5, 7, 4), (9, 4, 7, 8, 10, 5), (6, 11, 7, 1, 2, 9), (11, 5, 7, 9, 4, 2)]
            self.taskname = self.taskname + '_missing_all_6'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        elif self.missing_1_12:
            self.specific_training_leads = [(5,), (10, 2), (2, 9, 10), (0, 9, 7, 3), (3, 9, 5, 8, 10), 
                                            (1, 7, 8, 5, 2, 10), (4, 0, 6, 11, 10, 5, 3), (4, 1, 3, 10, 9, 11, 0, 2), (3, 2, 8, 10, 0, 1, 9, 4, 6), (10, 3, 0, 1, 8, 9, 11, 7, 4, 6), 
                                            (0, 1, 3, 8, 7, 9, 5, 6, 2, 10, 4), (8, 7, 1, 5, 2, 11, 6, 3, 0, 4, 9, 10), (8,), (3, 7, 1, 11, 10), (10, 1), 
                                            (3, 0, 7, 4, 1, 10, 9, 6, 11, 2), (1, 9, 2, 0, 8, 11), (6, 11, 10, 4, 5, 0), (4, 3), (6, 2, 7, 3, 0, 1)]
            self.taskname = self.taskname + '_missing_1_12'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        elif self.missing_7_12:
            self.specific_training_leads = [(4, 1, 11, 3, 0, 9, 7), (7, 5, 8, 10, 3, 0, 1, 2), (1, 4, 11, 3, 2, 9, 10, 6, 0), (6, 2, 8, 9, 10, 5, 0, 4, 3, 1), (0, 7, 3, 6, 11, 10, 9, 8, 2, 5, 4), 
                                            (5, 2, 6, 9, 11, 0, 1, 4, 8, 10, 7, 3), (0, 5, 8, 11, 7, 6, 2, 3), (4, 9, 5, 2, 7, 8, 0, 3, 11), (9, 3, 7, 5, 1, 6, 4, 8, 10), (11, 1, 5, 0, 10, 2, 7), 
                                            (11, 10, 0, 4, 6, 8, 3, 1), (5, 8, 7, 3, 0, 9, 11, 10, 2, 1), (11, 1, 7, 6, 2, 4, 3, 0, 10, 9, 5), (7, 1, 4, 6, 11, 2, 9), (7, 5, 3, 9, 0, 2, 8, 1, 10), 
                                            (3, 4, 10, 1, 11, 2, 7), (0, 11, 5, 9, 2, 3, 7, 8), (5, 7, 1, 4, 9, 0, 10, 2, 6, 11), (10, 4, 6, 3, 1, 8, 7, 0, 9, 11), (10, 11, 3, 2, 1, 8, 7, 9)]
            self.taskname = self.taskname + '_missing_7_12'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        elif self.missing_1_6:
            self.specific_training_leads = [(10,), (9, 3), (7, 3, 4), (4, 3, 6, 11), (0, 11, 9, 5, 10), 
                                            (5, 6, 4, 10, 7, 9), (4,), (9, 1, 11, 8, 5, 6), (1, 10, 9, 7), (3, 6, 1), 
                                            (1, 8, 11, 7, 4), (7, 10, 9, 4), (4, 5, 10), (5, 3, 1, 6), (0, 4), 
                                            (11, 6, 2, 7, 9, 10), (0, 7, 6), (7,), (0, 9), (11, 1, 4, 0, 9, 7)]
            self.taskname = self.taskname + '_missing_1_6'
            self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    # def local_holdout(self, local_datas, shuffle=False):
    #     """split each local dataset into train data and valid data according the rate."""
    #     train_cidxs = []
    #     valid_cidxs = []
    #     for local_data in local_datas:
    #         if shuffle:
    #             np.random.shuffle(local_data)
    #         k = int(len(local_data) * (1-self.local_holdout_rate))
    #         train_cidxs.append(local_data[:k])
    #         valid_cidxs.append(local_data[k:])
    #     return train_cidxs, valid_cidxs

    def load_data(self):
        self.train_data = PTBXL_REDUCE_Dataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=True
        )
        self.test_data = PTBXL_REDUCE_Dataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=False
        )
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads = 12
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads, contrastive_weight=0):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        # import pdb; pdb.set_trace()
        loss = model(tdata[0], tdata[-1], leads, contrastive_weight)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, leads, contrastive_weight=0, batch_size=64, num_workers=0):
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
            # import pdb; pdb.set_trace()
            
            loss, outputs = model(batch_data[0], batch_data[-1], leads, contrastive_weight)
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
            
        
        # loss_eval = [loss / (batch_id + 1) for loss in total_loss]
        loss_eval = [loss for loss in total_loss]
        
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
    def server_test(self, model, dataset, leads, contrastive_weight=0, batch_size=64, num_workers=0):
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
            # loss_each_modal = [[] for i in range(self.n_leads)]
            # loss_each_modal = [0]*12
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                loss, outputs = model(batch_data[0], batch_data[-1], leads[test_combi_index], contrastive_weight)
                # for i in range(self.n_leads):
                #     loss_each_modal[i] += loss_leads[i] * len(batch_data[-1])
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)
            result['acc'+str(test_combi_index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        # import pdb;pdb.set_trace()
        return result


    @torch.no_grad()
    def full_modal_server_test(self, model, dataset, leads, contrastive_weight=0, batch_size=64, num_workers=0):
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
            loss, outputs = model(batch_data[0], batch_data[-1], leads, contrastive_weight)
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