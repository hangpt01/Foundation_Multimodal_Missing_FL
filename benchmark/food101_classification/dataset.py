from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import torch
import random
from PIL import Image
import pandas as pd
from transformers import ViltProcessor

class Food101Dataset(Dataset):
    def __init__(self, root, download=False, train=True, subset=False):
        super(Food101Dataset, self).__init__()
        self.root = root
        
        # if not os.path.exists(self.root):
        #     print("Download dataset ...")
        #     os.system('bash ./benchmark/food101_classification/download.sh')
        
        # self.stop_chars = ["#","|","%","@","&",";",".com",":","\\","/",">","<","=","{","}"]
        self.train = train 
        self.mode = 'train' if train else 'test'
        self.x = torch.load(os.path.join(self.root, f'{self.mode}_inputs.pt'))       
        self.y = torch.load(os.path.join(self.root, f'{self.mode}_labels.pt'))       
        self.y = np.array(self.y)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        input = self.x[index]
        label = self.y[index]
        
        # import pdb; pdb.set_trace()
        return input, torch.tensor(label)
    
    
def collate(batch):
    # import pdb; pdb.set_trace()
    batch_ = batch
    batch = []
    labels = []
    for input, label in batch_:
        batch.append(input)
        labels.append(label)
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'][0] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    # import pdb; pdb.set_trace()
    # labels = [item['labels'].item() for item in batch]

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    
    # import pdb; pdb.set_trace()
    batch = {}
    batch['input_ids'] = torch.cat(input_ids)
    batch['attention_mask'] = torch.cat(attention_mask)
    batch['token_type_ids'] = torch.cat(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    # batch['labels'] = torch.tensor(labels)

    return batch, torch.tensor(labels)   


        
if __name__=='__main__':
    my_dataset = Food101Dataset(root='./benchmark/RAW_DATA/FOOD101', train=True)
    
    data_loader = DataLoader(my_dataset, batch_size=4, shuffle=True, collate_fn=collate) # len: 61127 - batch1; 1529 0 batch 40
    batch, labels = next(iter(data_loader))
    print(labels.shape)
    for k,v in batch.items():
        print(k, v.shape)
                                                                                                    # test:22716                    
# Iterate over the dataset and print a sample
    # for batch in data_loader:
    #     # sample = batch[0]  # Assuming batch size is 1
    #     print(batch)
    #     import pdb; pdb.set_trace()
    
    # subset = Food101Subset(dataset, range(100))
    # subset.local_missing_setup([0, 1], 0.2, 0.2)
    # print(len(dataset))
    # x, y = dataset[0]
    # print(x[1], y)
    
    # labels = np.array(dataset.text_labels['label'].unique(), dtype=object)
    # permutation = np.random.permutation(np.where(dataset.text_labels['label']=='ceviche')[0])
    # split = np.array_split(permutation, 20)
    # print(split)
        
        