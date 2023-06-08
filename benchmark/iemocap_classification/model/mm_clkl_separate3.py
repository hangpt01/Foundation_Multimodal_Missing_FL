import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from itertools import chain, combinations
import numpy as np

# 1-modality Extractor

class TextExtractor(FModule):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=1, batch_first=True)
        self.mapping_network = nn.Linear(200, 512)

    def forward(self, x):  
        batch = len(x)
        x_ = self.encoder_layer(x)
        x = torch.concat((x, x_), dim=2)
        output = self.mapping_network(x)
        return output.flatten(start_dim=1)

class AudioExtractor(FModule):
    def __init__(self):
        super(AudioExtractor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=1, batch_first=True)
        self.mapping_network = nn.Linear(200, 512)

    def forward(self, x):  
        batch = len(x)
        x_ = self.encoder_layer(x)
        x = torch.concat((x, x_), dim=2)
        output = self.mapping_network(x)
        return output.flatten(start_dim=1)

class VisionExtractor(FModule):
    def __init__(self):
        super(VisionExtractor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=1, batch_first=True)
        self.mapping_network = nn.Linear(1024, 512)

    def forward(self, x):  
        batch = len(x)
        x_ = self.encoder_layer(x)
        # import pdb; pdb.set_trace()
        x = torch.concat((x, x_), dim=2)
        output = self.mapping_network(x)
        return output.flatten(start_dim=1)


class VisionEncoder(FModule):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.vision_extractor = VisionExtractor()
        self.projector = nn.Linear(512, 256, True)
    def forward(self, vision):
        vision = self.vision_extractor(vision)
        return self.projector(vision)

class JointEncoder(FModule):
    def __init__(self):
        super(JointEncoder, self).__init__()
        self.text_extractor = TextExtractor()
        self.audio_extractor = AudioExtractor()
        self.vision_extractor = VisionExtractor()
        self.projector = nn.Linear(512, 256, True)
    def forward(self, text, audio, vision):
        text = self.text_extractor(text)
        audio = self.audio_extractor(audio)
        vision = self.vision_extractor(vision)
        # return self.projector(torch.concat((text, vision), dim=1))
        return self.projector(text+audio+vision)
    
    

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(256, 6, True)
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
        self.modalities = ["text", "audio", "vision"]
        
        # encoder
        self.vision_encoder = VisionEncoder()
        self.joint_encoder = JointEncoder()

        # classifier
        self.vision_classifier = Classifier()
        self.joint_classifier = Classifier()

        # criterion
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, samples, labels, modalities, contrastive_weight=0.0, temperature=1.0, kl_weight=0, name=None, iter=None):
        # import pdb; pdb.set_trace()
        if modalities == ["text", "audio", "vision"]:
            joint_encoding = self.joint_encoder(samples["text"], samples["audio"], samples["vision"])
            joint_outputs = self.joint_classifier(joint_encoding)
            joint_loss = self.CELoss(joint_outputs, labels)

            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_classifier(vision_encoding)
            vision_loss = self.CELoss(vision_outputs, labels)

            # if kl_weight > 0:
            #     joint_encoding_copy = joint_encoding.clone().detach()
            #     kl_loss = multivariate_KL_divergence(joint_encoding_copy, vision_encoding, joint_encoding.device)
            #     vision_loss += kl_weight*kl_loss

            # elif contrastive_weight > 0:
            #     device = joint_encoding.device
            #     batch_size = labels.shape[0]
            #     contrastive_loss = 0.0
            #     joint_encoding_copy = joint_encoding.clone().detach()
            #     # vision_encoding_copy = vision_encoding.clone().detach()
            #     vision_encoding_copy = vision_encoding
            #     # concat_reprs = torch.concat((joint_encoding_copy, vision_encoding_copy), dim=0)
            #     # concat_reprs = torch.concat((joint_encoding_copy, vision_encoding_copy), dim=0)
            #     exp_sim_matrix = torch.exp(torch.mm(concat_reprs, concat_reprs.t().contiguous()) / temperature)
            #     mask = (torch.ones_like(exp_sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
            #     exp_sim_matrix = exp_sim_matrix.masked_select(mask=mask).view(2 * batch_size, -1)
            #     positive_exp_sim = torch.exp(torch.sum(joint_encoding_copy * vision_encoding_copy, dim=-1) / temperature)
            #     positive_exp_sim = torch.concat((positive_exp_sim, positive_exp_sim), dim=0)
            #     contrastive_loss += - torch.log(positive_exp_sim / exp_sim_matrix.sum(dim=-1))
            #     vision_loss += contrastive_weight * contrastive_loss.mean()

            return joint_loss, joint_outputs, vision_loss, vision_outputs
        elif modalities == ["vision"]:
            vision_encoding = self.vision_encoder(samples["vision"])
            vision_outputs = self.vision_classifier(vision_encoding)
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