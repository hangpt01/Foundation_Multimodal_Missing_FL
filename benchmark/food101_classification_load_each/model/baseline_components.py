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

    def forward(self, input, backbone):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        token_type_ids = input['token_type_ids']
        pixel_values = input['pixel_values']
        pixel_mask = input['pixel_mask']
        head_mask = None
        inputs_embeds = None
        image_embeds = None
        image_token_type_idx = None
        output_attentions = None
        output_hidden_states  = None
        return_dict = True


        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #     self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")
        
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
        sequence_output = backbone.layernorm(sequence_output)
        pooled_output = backbone.pooler(sequence_output) if backbone.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        # import pdb; pdb.set_trace()
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


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
        
        # self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.backbone_components = ViltComponents()
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, backbone, batch, labels, leads):
        # import pdb; pdb.set_trace()
        missing_batch = dict()
        batch_size = labels.shape[0]
        device = labels.device
        for k,v in batch.items():
            if leads == [0] and k == 'input_ids':
                v = torch.tensor([101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device).repeat(batch_size, 1)
            if leads == [0] and k == 'attention_mask':
                v = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device).repeat(batch_size, 1)
            if leads == [1] and k == 'pixel_values':
                # import pdb; pdb.set_trace()
                v = torch.ones(v.shape, device=device)
            missing_batch[k] = v
            # import pdb; pdb.set_trace()

        # features = backbone(**missing_batch)
        features = self.backbone_components(missing_batch, backbone)
        # import pdb; pdb.set_trace()

        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()