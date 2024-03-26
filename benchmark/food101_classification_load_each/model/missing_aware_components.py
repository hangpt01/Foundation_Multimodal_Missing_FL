from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
import numpy as np
from utils.fmodule import FModule

class ViltComponents(FModule):
    def __init__(self):
        super().__init__()

    def forward(self, input, prompt, backbone, batch_size, device):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        token_type_ids = input['token_type_ids']
        pixel_values = input['pixel_values']
        pixel_mask = input['pixel_mask']
        inputs_embeds = None
        image_embeds = None
        image_token_type_idx = None,
        output_attentions = None,
        output_hidden_states  = None,
        return_dict = None,

        # text_batch_size, seq_length = input_shape
        # image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = backbone.get_head_mask(head_mask, backbone.config.num_hidden_layers)

        embedding_output, attention_mask = backbone.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = backbone.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = backbone.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class TextPrompt(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        num_context_vectors = 16
        embed_dim = 40
        ctx_vectors = torch.empty(num_context_vectors, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)
        self.dim_reduce_input_ids = nn.Linear(40*2, 40)
        self.dim_reduce_att_mask = nn.Linear(40*2, 40)

        
    def forward(self, batch, batch_size, processor, device):
        new_batch = batch.copy()
        prompt = self.prompt_learner
        
        dummy_image = torch.ones((batch_size,224,224))

        prompt_input_ids = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['input_ids']
        input_ids_concat = torch.cat((prompt_input_ids, batch['input_ids']), dim=1)
        new_batch['input_ids'] = self.dim_reduce_input_ids(input_ids_concat)
        
        prompt_attention_mask = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['attention_mask']
        attention_mask_concat = torch.cat((prompt_attention_mask, batch['attention_mask']), dim=1)
        new_batch['attention_mask'] = self.dim_reduce_att_mask(attention_mask_concat)

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
        
    def forward(self, batch, batch_size, processor):
        new_batch = batch.copy()
        prompt = self.prompt_learner
        
        dummy_image = torch.ones((batch_size,224,224))

        # No text -> add prompt as input text
        new_batch['input_ids'] = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['input_ids']
        new_batch['attention_mask'] = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['attention_mask']

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
        
    def forward(self, batch, batch_size, processor):
        new_batch = batch.copy()
        prompt = self.prompt_learner
        
        dummy_image = torch.ones((batch_size,224,224))

        prompt_input_ids = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['input_ids']
        input_ids_concat = torch.cat((prompt_input_ids, batch['input_ids']), dim=1)
        new_batch['input_ids'] = self.dim_reduce(input_ids_concat)
        
        prompt_attention_mask = processor(dummy_image, prompt, padding="max_length", truncation=True, max_length=40, return_tensors="pt")['attention_mask']
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
        # self.processor = BertTokenizer.from_pretrained('bert-base-uncased')

        # self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        # for param in self.processor.parameters():
        #     param.requires_grad = False
        self.text_prompt = TextPrompt()
        self.image_prompt = ImagePrompt()
        self.complete_prompt = CompletePrompt()
        
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, backbone, batch, labels, leads):
        # import pdb; pdb.set_trace()
        missing_batch = dict()
        device = labels.device
        batch_size = labels.shape[0]
        if leads == [0]:          # image-only
            missing_batch = self.image_prompt(batch, batch_size, self.processor)
        elif leads == [1]:        # text-only
            missing_batch = self.text_prompt(batch, batch_size, self.processor, device)
        else: 
            missing_batch = self.complete_prompt(batch, batch_size, self.processor)

        features = backbone(**missing_batch)
        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        # import pdb; pdb.set_trace()
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()