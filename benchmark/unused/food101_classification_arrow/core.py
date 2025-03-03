import functools
from .dataset import FOOD101Dataset
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
from transformers import (
    ViltProcessor,
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
import numpy as np
from tqdm import tqdm
import warnings
import collections
warnings.filterwarnings('ignore')
    
class TaskPipe(IDXTaskPipe):
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        
        origin_class = getattr(importlib.import_module(class_path), class_name)
        # import pdb; pdb.set_trace()
        data_dir = "./benchmark/RAW_DATA/FOOD101/generate_arrows"
        transform_keys = ['pixelbert']
        split="train"
        image_size = 384
        max_text_len = 40
        draw_false_image = 0
        draw_false_text = 0
        image_only = False
        _config = {
            'missing_ratio':
                {'test': 0.7,
                'train': 0.7},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables/',
            'missing_type':
                {'test': 'both',
                'train': 'both'},
            'both_ratio': 0.5,
            'simulate_missing': False
        }
        missing_info = {
                'ratio' : _config["missing_ratio"],
                'type' : _config["missing_type"],
                'both_ratio' : _config["both_ratio"],
                'missing_table_root': _config["missing_table_root"],
                'simulate_missing' : _config["simulate_missing"]
            }        
        
        # origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        # origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        # import pdb; pdb.set_trace()
        collator = DataCollatorForLanguageModeling

        origin_train_data = FOOD101Dataset(data_dir, transform_keys, split='train', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
        origin_train_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        # origin_train_data.mlm_collator = collator(tokenizer=origin_train_data.tokenizer, mlm=True, mlm_probability=0.15)
        # origin_train_data.collate = functools.partial(origin_train_data.collate, mlm_collator=origin_train_data.mlm_collator)

        origin_test_data = FOOD101Dataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
        origin_test_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        # origin_test_data.mlm_collator = collator(tokenizer=origin_test_data.tokenizer, mlm=True, mlm_probability=0.15)
        # origin_test_data.collate = functools.partial(origin_test_data.collate, mlm_collator=origin_test_data.mlm_collator)
        # import pdb; pdb.set_trace()
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = []
        valid_datas = []
        # modalities_list = []
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
            # modalities_list.append(feddata[name]['modalities'])
            # modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names']

def collate(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert (
            len(size) == 3
        ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, max_height, max_width)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask
            # import pdb; pdb.set_trace()
            
    return dict_batch


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
        # import pdb; pdb.set_trace()
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
    labels = np.unique(generator.train_data.labels)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.labels == label)[0])
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

# def get_food101_loader(dataset, batch_size=40, shuffle=True, num_workers=8, vocab_size=30522):
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate)
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0, missing=False):
        super(TaskGen, self).__init__(benchmark='food101_classification_arrow',
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
        self.visualize=self.visualize_by_class
        # import pdb; pdb.set_trace()
        # self.rawdata_path = os.path.join(self.rawdata_path, str(self.num_classes)+'_classes')
        self.source_dict = {
            'class_path': 'benchmark.food101_classification_arrow.dataset',
            'class_name': 'FOOD101Dataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train': 'True',
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train':' False'
            }
        }
        self.data_dir = os.path.join(self.rawdata_path, 'generate_arrows')
        _config = {
            'missing_ratio':
                {'test': 0.7,
                'train': 0.7},
            'missing_table_root': './benchmark/RAW_DATA/FOOD101/missing_tables/',
            'missing_type':
                {'test': 'both',
                'train': 'both'},
            'both_ratio': 0.5,
            'simulate_missing': False
        }
        self.missing_info = {
                'ratio' : _config["missing_ratio"],
                'type' : _config["missing_type"],
                'both_ratio' : _config["both_ratio"],
                'missing_table_root': _config["missing_table_root"],
                'simulate_missing' : _config["simulate_missing"]
            }
        self.transform_keys = ['pixelbert']
        self.image_size = 384
        self.max_text_len = 40
        self.draw_false_image = 0
        self.draw_false_text = 0
        self.image_only = False

        self.missing=missing
        self.local_holdout_rate = 0.1
        # self.specific_training_leads = None
        
        if self.missing and self.num_clients==20:
            # self.specific_training_leads = [[0, 1]]*10 + [[0]]*5 + [[1]]*5 
            self.taskname = self.taskname + '_missing_each_0.25'
            # self.taskname = self.taskname + '_train_test_missing_both_0.7'
        if self.missing and self.num_clients==10:
            self.specific_training_leads = [[0, 1]]*6 + [[0]]*2 + [[1]]*2 
            self.taskname = self.taskname + '_missing_each_0.2'  
        if self.missing and self.num_clients==1:
            self.specific_training_leads = [[0,1]]
            self.taskname = self.taskname + '_centralized_no_missing'
        
        # self.taskname = self.taskname + '_' + str(self.num_classes) + '_classes'    
        
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
        collator = DataCollatorForLanguageModeling

        self.train_data = FOOD101Dataset(self.data_dir, self.transform_keys, split='train', 
                                image_size=self.image_size,
                                max_text_len=self.max_text_len,
                                draw_false_image=self.draw_false_image,
                                draw_false_text=self.draw_false_text,
                                image_only=self.image_only,
                                missing_info=self.missing_info)
        self.train_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        self.train_data.mlm_collator = collator(tokenizer=self.train_data.tokenizer, mlm=True, mlm_probability=0.15)
        self.train_data.collate = functools.partial(self.train_data.collate, mlm_collator=self.train_data.mlm_collator)
        # import pdb; pdb.set_trace()
        self.test_data = FOOD101Dataset(self.data_dir, self.transform_keys, split='test', 
                                image_size=self.image_size,
                                max_text_len=self.max_text_len,
                                draw_false_image=self.draw_false_image,
                                draw_false_text=self.draw_false_text,
                                image_only=self.image_only,
                                missing_info=self.missing_info)
        self.test_data.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        self.test_data.mlm_collator = collator(tokenizer=self.test_data.tokenizer, mlm=True, mlm_probability=0.15)
        self.test_data.collate = functools.partial(self.test_data.collate, mlm_collator=self.test_data.mlm_collator)
        
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
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads=2
        self.DataLoader = DataLoader
        
    def get_data_loader(self, dataset, batch_size=40, shuffle=True, num_workers=8, vocab_size=30522):
        # import pdb; pdb.set_trace()
        collator = DataCollatorForLanguageModeling
        tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        mlm_collator = collator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=functools.partial(collate, mlm_collator=mlm_collator))
        # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def data_to_device(self, data):
        # for k, v in data.items():
        #     print(k,len(v))
        batch = data
        # import pdb; pdb.set_trace()
        batch['image'][0] = batch['image'][0].to(self.device)
        for key in ['text_ids', 'text_labels', 'text_ids_mlm', 'text_labels_mlm', 'text_masks']:
            new_ls = []
            for tensor in data[key]:
                new_ls.append(tensor.to(self.device)) 
            batch[key] = torch.stack(new_ls)
        # batch = {k:v.to(self.device) for k,v in data.items()}
        # import pdb; pdb.set_trace()
        return batch
        
    def train_one_step(self, model, transformer, text_embeddings, data):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        # import pdb; pdb.set_trace()

        batch = self.data_to_device(data)
        # import pdb; pdb.set_trace()
        model.to(self.device) # y.device
        transformer.to(self.device)
        text_embeddings.to(self.device)
        # print(tdata[0])
        loss, _ = model(transformer, text_embeddings, batch)
        # backbone.to('cpu')
        # print(loss.cpu().item())
        return {'loss': loss}
    
    @torch.no_grad()
    def test(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            # import pdb; pdb.set_trace()
            loss, outputs = model(transformer, text_embeddings, batch_data)
            total_loss += loss.item()
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # TO_DELETE
            if batch_id==1:
                break
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / (batch_id+1),
            'acc': accuracy
        }
    

    @torch.no_grad()
    def evaluate(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
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
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            loss, outputs = model(transformer, text_embeddings, batch_data)
            if batch_id==0:
                total_loss = loss
            else:
                total_loss = loss + total_loss
            # TO_DELETE
            if batch_id==1:
                break
        loss_eval = loss / (batch_id + 1) 
        return loss_eval

    
    @torch.no_grad()
    def server_test(self, model, transformer, text_embeddings, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        # import pdb; pdb.set_trace()
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        result = dict() 
        # for test_combi_index in range(len(2)):
        total_loss = 0.0
        labels = list()
        predicts = list()   
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data['label'])
            loss, outputs = model(transformer, text_embeddings, batch_data)
            total_loss += loss.item() * len(batch_data['label'])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            # TO_DELETE
            if batch_id==1:
                break
        # import pdb; pdb.set_trace()
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        # for i in range(self.n_leads):
        #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
        result['loss'] = total_loss / len(dataset)
        result['acc'] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        # import pdb;pdb.set_trace()
        return result
        