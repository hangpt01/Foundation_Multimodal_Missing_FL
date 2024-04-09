from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import numpy as np
from utils.fmodule import FModule
import benchmark.food101_classification_arrow.model.vision_transformer_prompts as vit
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
        super(Model, self).__init__()
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

        self.hparams_config = {'batch_size': 32, 
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
                        'drop_rate': 0.1,
                        'max_image_len': 40,
                        'load_path': 'benchmark/food101_classification_arrow/pretrained_model_weight/vilt_200k_mlm_itm.ckpt'}
        
        self.device = None
        self.transformer = getattr(vit, self.hparams_config["vit"])(
            pretrained=False, config=self.hparams_config
        )
        # import pdb; pdb.set_trace()  

        self.pooler = Pooler(config["hidden_size"])
        self.pooler.apply(init_weights)

        # ===================== Downstream ===================== #
        ckpt = torch.load(self.hparams_config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        # since the pre-trained max_text_len is 40,
        # we upsample the weight of position embedding to determined max_text_len
        if config["max_text_len"] != 40:
            state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
            pos_emb = state_dict['text_embeddings.position_embeddings.weight']
            pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
            state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
        self.load_state_dict(state_dict, strict=False)

        hs = self.hparams_config["hidden_size"]

        cls_num = 101
        self.food101_classifier = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, cls_num),
        )
        self.food101_classifier.apply(init_weights)   
  
        self.prompt_type = self.hparams_config["prompt_type"]
        prompt_length = self.hparams_config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams_config["hidden_size"]
        self.learnt_p = self.hparams_config["learnt_p"]
        self.prompt_layers = self.hparams_config["prompt_layers"]
        self.multi_layer_prompt = self.hparams_config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1


        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1) 
       
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)            
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)            
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        set_metrics(self)
        self.current_tasks = list()

        self.records = {}
        
    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
            is_train=None,
        ):
        # import pdb; pdb.set_trace()
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # import pdb; pdb.set_trace()
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        self.device = img.device
        if image_embeds is None and image_masks is None:
                   
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams_config["max_image_len"],
                mask_it=mask_image,
            )
    

        else:
            patch_index, image_labels = (
                None,
                None,
            )
        # import pdb; pdb.set_trace()
        # (batch, 40, 768), (batch, 217, 768)
        text_embeds, image_embeds = (        
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)), # + (batch,40)
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)                  # + (batch,217)
            ),
        )
        
        # instance wise missing aware prompts
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt        
            elif batch["missing_type"][idx] == 1:
                # import pdb; pdb.set_trace()
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_img_prompt
                # 3 prompt: ([6, 16, 768])
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            
            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)      #  ([batch, 6, 16, 768])
        # import pdb; pdb.set_trace()     
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()      #torch.Size([batch, 96])
                # import pdb; pdb.set_trace()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()   
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)    # torch.Size([batch, 329]);     batch, 353=257+96
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # torch.Size([1, 233, 768])             batch, 257, 768
        # import pdb; pdb.set_trace()
        x = co_embeds.detach()      # torch.Size([1, 233, 768])     batch, 257, 768=text+img

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=prompts[:,self.prompt_layers.index(i)],      # batch, 16, 768
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)
        # import pdb; pdb.set_trace()
        # x: torch.Size([1, 329, 768])
        x = self.transformer.norm(x)    # x: torch.Size([1, 329, 768])
        
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]   # len([0, 1, 2, 3, 4, 5]) * 16
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )       # ([1, 40, 768]), ([1, 193, 768])
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)
            
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret


    def forward(self, batch):
        ret = dict()
        # ret.update(objectives.compute_food101(self, batch))      

        phase = "train" if self.training else "val"
        if phase == "train":
            infer = self.infer(batch, mask_text=False, mask_image=False)
        else:
            infer = self.infer(batch, mask_text=False, mask_image=False)

        imgcls_logits = self.food101_classifier(infer["cls_feats"])

        imgcls_labels = batch["label"]
        imgcls_labels = torch.tensor(imgcls_labels).to(self.device).long()
        imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

        ret = {
            "food101_loss": imgcls_loss,
            "food101_logits": imgcls_logits,
            "food101_labels": imgcls_labels,
        }

        loss = getattr(self, f"{phase}_food101_loss")(ret["food101_loss"])
        acc = getattr(self, f"{phase}_food101_accuracy")(
            ret["food101_logits"], ret["food101_labels"]
        )
        self.log(f"food101/{phase}/loss", loss)


        return ret

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()