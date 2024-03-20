from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltModel, ViltProcessor
import numpy as np
from utils.fmodule import FModule

class TextPrompt(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        num_context_vectors = 16
        embed_dim = 40
        ctx_vectors = torch.empty(num_context_vectors, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)
        self.dim_reduce = nn.Linear(40*2, 40)
        
    def forward(self, batch, processor):
        new_batch = batch.copy()
        device = batch.device
        prompt = self.prompt_learner
        
        prompt_input_ids = processor(prompt)['input_ids']
        input_ids_concat = torch.cat((prompt_input_ids, batch['input_ids']), dim=1)
        new_batch['input_ids'] = self.dim_reduce(input_ids_concat)
        
        prompt_attention_mask = processor(prompt)['attention_mask']
        attention_mask_concat = torch.cat((prompt_attention_mask, batch['attention_mask']), dim=1)
        new_batch['attention_mask'] = self.dim_reduce(attention_mask_concat)

        # No image
        new_batch['pixel_values'] = torch.ones(batch['pixel_values'].shape, device=device)

        return new_batch
    
    
class ImagePrompt(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        num_context_vectors = 16
        embed_dim = 40
        ctx_vectors = torch.empty(num_context_vectors, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)
        # self.dim_reduce = nn.Linear(40*2, 40)
        
    def forward(self, batch, processor):
        new_batch = batch.copy()
        prompt = self.prompt_learner
        
        # No text -> add prompt as input text
        new_batch['input_ids'] = processor(text=prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['input_ids']
        new_batch['attention_mask'] = processor(text=prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['attention_mask']

        return new_batch

class CompletePrompt(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        num_context_vectors = 16
        embed_dim = 40
        ctx_vectors = torch.empty(num_context_vectors, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)
        self.dim_reduce = nn.Linear(40*2, 40)
        
    def forward(self, batch, processor):
        new_batch = batch.copy()
        device = batch.device
        prompt = self.prompt_learner
        
        prompt_input_ids = processor(prompt)['input_ids']
        input_ids_concat = torch.cat((prompt_input_ids, batch['input_ids']), dim=1)
        new_batch['input_ids'] = self.dim_reduce(input_ids_concat)
        
        prompt_attention_mask = processor(prompt)['attention_mask']
        attention_mask_concat = torch.cat((prompt_attention_mask, batch['attention_mask']), dim=1)
        new_batch['attention_mask'] = self.dim_reduce(attention_mask_concat)

        return new_batch

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
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        # self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        for param in self.processor.parameters():
            param.requires_grad = False
        self.text_prompt = TextPrompt()
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, backbone, batch, labels, leads):
        # import pdb; pdb.set_trace()
        missing_batch = dict()
        batch_size = labels.shape[0]
        device = labels.device
        # for k,v in batch.items():
        #     if leads == [0] and k == 'input_ids':
        #         v = torch.tensor([101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device).repeat(batch_size, 1)
        #     if leads == [0] and k == 'attention_mask':
        #         v = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device).repeat(batch_size, 1)
        #     if leads == [1] and k == 'pixel_values':
        #         # import pdb; pdb.set_trace()
        #         v = torch.ones(v.shape, device=device)
        #     missing_batch[k] = v
            # import pdb; pdb.set_trace()
        if leads == [1]:        # text-only
            missing_batch = self.text_prompt(batch, self.processor)
        else:
            missing_batch = batch
        features = backbone(**missing_batch)
        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        import pdb; pdb.set_trace()
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()