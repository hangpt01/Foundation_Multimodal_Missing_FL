from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
import numpy as np
from utils.fmodule import FModule

class Prompt_Learner(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # num_layer = 6
        num_context_vectors = 16
        embed_dim = 768
        # ctx_vectors = torch.empty(num_layer, num_context_vectors, embed_dim)
        ctx_vectors = torch.empty(num_context_vectors, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)

    def forward(self, device):
        return self.prompt_learner.to(device)

class ViltComponents(FModule):
    def __init__(self):
        super().__init__()
        self.prompt_length = 16
        self.prompt_layers = 1

    def forward(self, input, prompts, backbone):
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

        input_shape = input_ids.size()      # (batch, 40)
        head_mask = backbone.get_head_mask(head_mask, backbone.config.num_hidden_layers)

        embedding_output, attention_mask = backbone.embeddings(     # (batch, 269, 768), (batch, 269)
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

        # import pdb; pdb.set_trace()
        # Prompt - num_layer does not have meaning
        # embedding_output.element_size() * embedding_output.nelement() / (1024 * 1024)
        prompt_masks = torch.ones(input_shape[0], self.prompt_length*self.prompt_layers, dtype=prompts.dtype, device=prompts.device).unsqueeze(1).unsqueeze(1)
        # extended_prompt_mask = backbone.get_extended_attention_mask(prompt_masks, input_shape)
        extended_attention_mask = torch.cat([extended_attention_mask, prompt_masks], dim=3)    # (batch, 1, 1, 269+16)
        # extended_attention_mask
        embedding_output = torch.cat([embedding_output, prompts.unsqueeze(0).repeat(input_shape[0], 1, 1)], dim=1)       # (batch, 269+16, 768)

        # import pdb; pdb.set_trace()

        encoder_outputs = backbone.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )   
        # import pdb; pdb.set_trace()

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
    
        self.image_prompt = Prompt_Learner()
        self.text_prompt = Prompt_Learner()
        self.complete_prompt = Prompt_Learner()

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
        # prompts = [self.image_prompt, self.text_prompt, self.complete_prompt]
        self.image_prompt = self.image_prompt.to(device)
        self.text_prompt = self.text_prompt.to(device)
        self.complete_prompt = self.complete_prompt.to(device)

        if leads == [0]:
            features = self.backbone_components(missing_batch, self.image_prompt, backbone)
        elif leads == [1]:
            features = self.backbone_components(missing_batch, self.text_prompt, backbone)
        else:
            features = self.backbone_components(missing_batch, self.complete_prompt, backbone)
        # import pdb; pdb.set_trace()

        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        
        loss = self.criterion(outputs, labels.type(torch.int64))

        labels = labels.cpu().numpy().astype(np.int64)
        unique_labels = np.unique(labels)
        norm_features = F.normalize(self.complete_prompt, p=2, dim=1)
        contrative_loss = 0.0
        count = 0
        missing_prompt = [self.image_prompt, self.text_prompt]
        for prompt in missing_prompt:
            norm_prompt = F.normalize(prompt, p=2, dim=1)
            simi_mat = norm_features.matmul(norm_prompt.T)
            exp_simi_mat = torch.exp(simi_mat / 1.0)
            for label in unique_labels:
                positive_idx = np.where(labels == label)[0]
                negative_idx = np.where(labels != label)[0]
                positive = exp_simi_mat.diagonal()[positive_idx]
                negative = exp_simi_mat[positive_idx, :][:, negative_idx].sum(dim=1)
                contrative_loss -= torch.log(positive / (positive + negative)).sum()
                negative = exp_simi_mat[negative_idx, :][:, positive_idx].sum(dim=0)
                contrative_loss -= torch.log(positive / (positive + negative)).sum()
                count += positive_idx.shape[0] * 2
        # import pdb; pdb.set_trace()
        if count > 0:
            loss += 5 * contrative_loss / count

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()