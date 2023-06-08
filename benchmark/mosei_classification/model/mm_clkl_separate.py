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


def multivariate_KL_divergence(teacher_batch_input, student_batch_input, device):
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.to(device).unsqueeze(1)
    student_batch_input = student_batch_input.to(device).unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm.flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl  


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

    def forward(self, samples, labels, modalities, contrastive_weight=0.0, temperature=1.0, kl_weight=0, name=None, iter=None):
        if modalities == ["text", "vision"]:
            joint_encoding = self.joint_encoder(samples["text"], samples["vision"])
            joint_outputs = self.joint_regressor(joint_encoding)
            joint_loss = self.L1Loss(joint_outputs, labels)

            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_regressor(vision_encoding)
            vision_loss = self.L1Loss(vision_outputs, labels)

            if kl_weight > 0:
                joint_encoding_copy = joint_encoding.clone().detach()
                kl_loss = multivariate_KL_divergence(joint_encoding_copy, vision_encoding, joint_encoding.device)
                vision_loss += kl_weight*kl_loss

            elif contrastive_weight > 0:
                device = joint_encoding.device
                batch_size = labels.shape[0]
                contrastive_loss = 0.0
                joint_encoding_copy = joint_encoding.clone().detach()
                # vision_encoding_copy = vision_encoding.clone().detach()
                vision_encoding_copy = vision_encoding
                concat_reprs = torch.concat((joint_encoding_copy, vision_encoding_copy), dim=0)
                exp_sim_matrix = torch.exp(torch.mm(concat_reprs, concat_reprs.t().contiguous()) / temperature)
                mask = (torch.ones_like(exp_sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
                exp_sim_matrix = exp_sim_matrix.masked_select(mask=mask).view(2 * batch_size, -1)
                positive_exp_sim = torch.exp(torch.sum(joint_encoding_copy * vision_encoding_copy, dim=-1) / temperature)
                positive_exp_sim = torch.concat((positive_exp_sim, positive_exp_sim), dim=0)
                contrastive_loss += - torch.log(positive_exp_sim / exp_sim_matrix.sum(dim=-1))
                vision_loss += contrastive_weight * contrastive_loss.mean()

            return joint_loss, joint_outputs, vision_loss, vision_outputs
        elif modalities == ["vision"]:
            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_regressor(vision_encoding)
            vision_loss = self.L1Loss(vision_outputs, labels)

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