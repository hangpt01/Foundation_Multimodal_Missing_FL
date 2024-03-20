from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltModel, ViltProcessor, BertTokenizerFast
import numpy as np
from ..util import clip
from ..util.simple_tokenizer import SimpleTokenizer
from utils.fmodule import FModule


def get_class_embedding():
    label_dict = {
        "apple_pie": 0,
        "baby_back_ribs": 1,
        "baklava": 2
    }
    

class PromptLearner(FModule):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.num_context_vectors = 16
        # class-specific context vectors
        ctx_vectors = torch.empty(num_classes, self.num_context_vectors, 40)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_learner = nn.Parameter(ctx_vectors)
        
        prompt_prefix = " ".join(["X"] * self.num_context_vectors)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.num_context_vectors}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized


        classnames = ["apple_pie", "baby_back_ribs" , "baklava"]
        _tokenizer = SimpleTokenizer()
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # import pdb; pdb.set_trace()
        # for p in prompts:
        #     print(p)
        # import pdb; pdb.set_trace()
        # tokenized_prompts = torch.cat([tokenizer.tokenize(p, return_tensors="pt") for p in prompts])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.num_classes = num_classes
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

    def forward(self):
        # input x: (batch, 40)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # append class to the end of the prompts
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )


        # concat to text


        return prompts


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
    
        self.prompt_learner = PromptLearner()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, backbone, batch, labels, leads):
        # import pdb; pdb.set_trace()
        prompts = self.prompt_learner()
        import pdb; pdb.set_trace()
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

        features = backbone(**missing_batch)
        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        
        loss = self.criterion(outputs, labels.type(torch.int64))

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()