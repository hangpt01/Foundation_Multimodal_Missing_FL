import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from itertools import chain, combinations
from benchmark.mosei_classification.model.transformer_networks import TransformerEncoder
import numpy as np

# 1-modality Extractor
class TextExtractor(FModule):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.input_dim = 300
        self.hidden_dim = 30
        self.latent_dim = 60
        self.timestep = 50
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=False)
        # self.projector = nn.Linear(self.hidden_dim*self.timestep, self.latent_dim)

    def forward(self, x):       # (snapshot:50,300)
        batch = len(x)
        # import pdb; pdb.set_trace()
        input = x.reshape(batch, self.timestep, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return output.flatten(start_dim=1)
    

class VisionExtractor(FModule):
    def __init__(self):
        super(VisionExtractor, self).__init__()
        self.input_dim = 35
        self.hidden_dim = 30
        self.latent_dim = 60
        self.timestep = 50
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=False)
        # self.projector = nn.Linear(self.hidden_dim*self.timestep, self.latent_dim)

    def forward(self, x):       # (snapshot:50,300)
        batch = len(x)
        # import pdb; pdb.set_trace()
        input = x.reshape(batch, self.timestep, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return output.flatten(start_dim=1)


class VisionEncoder(FModule):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.vision_extractor = VisionExtractor()
        self.projector = nn.Linear(1500, 60, True)
    def forward(self, vision):
        vision = self.vision_extractor(vision)
        return self.projector(vision)

class JointEncoder(FModule):
    def __init__(self):
        super(JointEncoder, self).__init__()
        self.text_extractor = TextExtractor()
        self.vision_extractor = VisionExtractor()
        self.projector = nn.Linear(1500*2, 60, True)
    def forward(self, text, vision):
        text = self.text_extractor(text)
        vision = self.vision_extractor(vision)
        return self.projector(torch.concat((text, vision), dim=1))
    

class Regressor(FModule):
    def __init__(self):
        super(Regressor, self).__init__()
        self.ln = nn.Linear(60, 1, True)
    def forward(self, x):
        return torch.squeeze(self.ln(x))
    

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.modalities = ["text", "vision"]
        
        # encoder
        self.vision_encoder = VisionEncoder()
        self.joint_encoder = JointEncoder()

        # regressor
        self.vision_regressor = Regressor()
        self.joint_regressor = Regressor()

        # criterion
        self.L1Loss = nn.L1Loss()

    def forward(self, samples, labels, modalities, contrastive_weight=0.0, temperature=1.0, name=None, iter=None):
        if modalities == ["text", "vision"]:
            joint_encoding = self.joint_encoder(samples["text"], samples["vision"])
            joint_outputs = self.joint_regressor(joint_encoding)
            joint_loss = self.CELoss(joint_outputs, labels)

            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_regressor(vision_encoding)
            vision_loss = self.CELoss(vision_outputs, labels)

            return joint_loss, joint_outputs, vision_loss, vision_outputs
        elif modalities == ["vision"]:
            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_regressor(vision_encoding)
            vision_loss = self.CELoss(vision_outputs, labels)

            return None, None, vision_loss, vision_outputs

if __name__ == '__main__':
    model = Model()
    # with open("./model.txt","w") as f:
    #     f.write(str(model))
    # print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()
    samples = {
        'text': torch.rand(size=(64, 50, 300)),
        'vision': torch.rand(size=(64, 50, 35))
    }
    labels = torch.rand(size=(64,))