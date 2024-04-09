from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import numpy as np
from utils.fmodule import FModule
import vision_transformer_prompts as vit
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from torchmetrics import Metric


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1)>1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits)>0.5).long()
            
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total    
    

def set_metrics(pl_module):
    split = 'train'
    k = 'food101'
    setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
    setattr(pl_module, f"{split}_{k}_loss", Scalar())       
                


class Model(FModule):
    def __init__(self):
        config = {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "max_text_len": 40,
            "drop_rate": 0.1
        }

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)

        self.hparams = {'batch_size': 32, 
                        'prompt_type': 'input', 
                        'prompt_length': 16, 
                        'learnt_p': True, 
                        'prompt_layers': [0, 1, 2, 3, 4, 5], 
                        'multi_layer_prompt': True, 
                        'max_text_len': 40, 
                        'vocab_size': 30522, 
                        'vit': 'vit_base_patch32_384', 
                        'hidden_size': 768, 
                        'num_heads': 12, 
                        'num_layers': 12, 
                        'load_path': 'pretrained_model_weight/vilt_200k_mlm_itm.ckpt'}
        
        self.transformer = getattr(vit, self.hparams.config["vit"])(
            pretrained=False, config=self.hparams.config
        )
        # import pdb; pdb.set_trace()  

        self.pooler = Pooler(config["hidden_size"])
        self.pooler.apply(init_weights)

        # ===================== Downstream ===================== #
        ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        # since the pre-trained max_text_len is 40,
        # we upsample the weight of position embedding to determined max_text_len
        if config["max_text_len"] != 40:
            state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
            pos_emb = state_dict['text_embeddings.position_embeddings.weight']
            pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
            state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
        self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        cls_num = 101
        self.food101_classifier = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, cls_num),
        )
        self.food101_classifier.apply(init_weights)   
  
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1


        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1) 

        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        set_metrics(self)
        self.current_tasks = list()

        self.records = {}
        
    def forward(self, batch):
        ret = dict()
        # Classification for Food101
        if "food101" in self.current_tasks:
            import pdb; pdb.set_trace()
            ret.update(objectives.compute_food101(self, batch))              

        return ret

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
        if leads == [0]:
            features = self.backbone_components(missing_batch, self.image_prompt(device), backbone)
        elif leads == [1]:
            features = self.backbone_components(missing_batch, self.text_prompt(device), backbone)
        else:
            features = self.backbone_components(missing_batch, self.complete_prompt(device), backbone)
        # import pdb; pdb.set_trace()

        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()