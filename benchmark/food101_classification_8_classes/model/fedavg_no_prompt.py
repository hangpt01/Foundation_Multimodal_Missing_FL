from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.fmodule import FModule


class Pooler(FModule):
    def __init__(self):
        super().__init__()
        hidden_size = 768
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
    

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        cls_num = 101
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, cls_num),
        )
    
    def forward(self, x):
        return self.classifier(x)


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.hparams_config = {'hidden_size': 768, 
                                'max_image_len': 40}
        
        self.device = None
        self.transformer = None
        self.text_embeddings = None
        # import pdb; pdb.set_trace()  

        self.token_type_embeddings = nn.Embedding(2, self.hparams_config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        self.pooler = Pooler()
        self.pooler.apply(init_weights)

        self.classifier = Classifier()
        self.classifier.apply(init_weights)   

        
    def infer(
            self,
            batch,
            transformer,
            text_embeddings,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
        ):
        self.transformer = transformer
        self.text_embeddings = text_embeddings
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

        # import pdb; pdb.set_trace()
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
        
        co_masks = torch.cat([text_masks, image_masks], dim=1)    # torch.Size([batch, 329]);     batch, 353=257+96
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # torch.Size([1, 233, 768])             batch, 257, 768
        # import pdb; pdb.set_trace()
        x = co_embeds.detach()      # torch.Size([1, 233, 768])     batch, 257, 768=text+img

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
        # import pdb; pdb.set_trace()
        # x: torch.Size([1, 329, 768])
        x = self.transformer.norm(x)    # x: torch.Size([1, 329, 768])

        
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )       # ([1, 40, 768]), ([1, 193, 768])
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


    def forward(self, transformer, text_embeddings, batch):
        infer = self.infer(batch, transformer, text_embeddings, mask_text=False, mask_image=False)
        imgcls_logits = self.classifier(infer["cls_feats"])

        imgcls_labels = batch["label"]
        imgcls_labels = torch.tensor(imgcls_labels).to(self.device).long()
        imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)
        # import pdb; pdb.set_trace()

        return imgcls_loss, imgcls_loss, imgcls_logits


if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()