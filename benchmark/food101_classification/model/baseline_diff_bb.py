from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltModel
import numpy as np
from utils.fmodule import FModule

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 101))
    
    def forward(self, x):
        return self.classifier(x)

    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.hidden_size = 768
        
        self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, batch, labels, leads):
        # import pdb; pdb.set_trace()
        
        features = self.backbone(**batch)
        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()