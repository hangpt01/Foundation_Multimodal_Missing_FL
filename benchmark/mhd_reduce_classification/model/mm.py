import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from itertools import chain, combinations

class ImageExtractor(FModule):
    def __init__(self):
        super(ImageExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return torch.flatten(x, start_dim=1)
    
class SoundExtractor(FModule):
    def __init__(self):
        super(SoundExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.flatten(x, start_dim=1)
    
class TrajectoryExtractor(FModule):
    def __init__(self):
        super(TrajectoryExtractor, self).__init__()
        self.ln1 = nn.Linear(200, 512)
        self.ln2 = nn.Linear(512, 512)
    def forward(self, x):
        x = F.silu(self.ln1(x))
        x = F.silu(self.ln2(x))
        return torch.flatten(x, start_dim=1)
    
class ImageProjector(FModule):
    def __init__(self):
        super(ImageProjector, self).__init__()
        self.ln = nn.Linear(128 * 7 * 7, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundProjector(FModule):
    def __init__(self):
        super(SoundProjector, self).__init__()
        self.ln = nn.Linear(2048, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class TrajectoryProjector(FModule):
    def __init__(self):
        super(TrajectoryProjector, self).__init__()
        self.ln = nn.Linear(512, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundProjector(FModule):
    def __init__(self):
        super(ImageSoundProjector, self).__init__()
        self.ln = nn.Linear(128 * 7 * 7 + 2048, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageTrajectoryProjector(FModule):
    def __init__(self):
        super(ImageTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(128 * 7 * 7 + 512, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundTrajectoryProjector(FModule):
    def __init__(self):
        super(SoundTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(2048 + 512, 64, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundTrajectoryProjector(FModule):
    def __init__(self):
        super(ImageSoundTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(128 * 7 * 7 + 2048 + 512, 64, True)
    def forward(self, x):
        return self.ln(x)

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(64, 10, True)
    def forward(self, x):
        return self.ln(x)

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.modalities = ["image", "trajectory"]
        self.combin = "+".join(self.modalities)
        # self.modalities = ["image", "sound", "trajectory"]

        # feature extractors
        self.feature_extractors = nn.ModuleDict({
            "image": ImageExtractor(),
            "sound": SoundExtractor(),
            "trajectory": TrajectoryExtractor()
        })
        
        # projectors
        self.projectors = nn.ModuleDict({
            "image": ImageProjector(),
            "sound": SoundProjector(),
            "trajectory": TrajectoryProjector(),
            "image+sound": ImageSoundProjector(),
            "image+trajectory": ImageTrajectoryProjector(),
            "sound+trajectory": SoundTrajectoryProjector(),
            "image+sound+trajectory": ImageSoundTrajectoryProjector()
        })

        # classifier
        self.classifier = Classifier()

        # criterion
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, samples, labels, contrastive_weight=0.0, temperature=1.0):
        current_modalities = list()
        batch_size = None
        device = None
        for modal in self.modalities:
            if modal in samples.keys():
                batch_size = samples[modal].shape[0]
                device = samples[modal].device
                current_modalities.append(modal)
        # import pdb; pdb.set_trace()
        if len(current_modalities) == 1:
            modal = current_modalities[0]
            features = self.feature_extractors[modal](samples[modal])
            representations = F.normalize(F.relu(self.projectors[modal](features)), p=2, dim=1)
            outputs = self.classifier(representations)
            loss = self.CELoss(outputs, labels)
        elif current_modalities == self.modalities:
            representations_dict = dict()
            features_dict = dict()
            for modal in self.modalities:
                features = self.feature_extractors[modal](samples[modal])
                features_dict[modal] = features
                representations_dict[modal] = F.normalize(F.relu(self.projectors[modal](features)), p=2, dim=1)
            joint_representations = F.normalize(F.relu(self.projectors[self.combin](torch.concat(tuple(features_dict.values()), dim=1))), p=2, dim=1)
            outputs = self.classifier(joint_representations)
            loss = self.CELoss(outputs, labels)
            if batch_size > 1 and contrastive_weight > 0.0:
                contrastive_loss = 0.0
                for modal in self.modalities:
                # for modal in ["trajectory"]:
                    concat_reprs = torch.concat((joint_representations, representations_dict[modal]), dim=0)
                    exp_sim_matrix = torch.exp(torch.mm(concat_reprs, concat_reprs.t().contiguous()) / temperature)
                    mask = (torch.ones_like(exp_sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
                    exp_sim_matrix = exp_sim_matrix.masked_select(mask=mask).view(2 * batch_size, -1)
                    positive_exp_sim = torch.exp(torch.sum(joint_representations * representations_dict[modal], dim=-1) / temperature)
                    positive_exp_sim = torch.concat((positive_exp_sim, positive_exp_sim), dim=0)
                    contrastive_loss += - torch.log(positive_exp_sim / exp_sim_matrix.sum(dim=-1))
                loss += contrastive_weight * contrastive_loss.mean()
        return loss, outputs
    
    def get_embedding(self, samples):
        features_dict = dict()
        representations_dict = dict()
        for modal in self.modalities:
            features = self.feature_extractors[modal](samples[modal])
            features_dict[modal] = features
            representations_dict[modal] = F.normalize(F.relu(self.projectors[modal](features)), p=2, dim=1)
        representations_dict[self.combin] = F.normalize(F.relu(self.projectors[self.combin](torch.concat(tuple(features_dict.values()), dim=1))), p=2, dim=1)
        return representations_dict

if __name__ == '__main__':
    model = Model()
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()