import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

# IMAGE_LATENT_DIM = 64
# SOUND_LATENT_DIM = 128
# TRAJECTORY_LATENT_DIM = 16
IMAGE_LATENT_DIM = 32
SOUND_LATENT_DIM = 32
TRAJECTORY_LATENT_DIM = 32
COMMON_DIM = 32

class ImageExtractor(FModule):
    def __init__(self):
        super(ImageExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.ln1 = nn.Linear(3136, 128, True)
        self.ln2 = nn.Linear(128, 128, True)
        self.ln3 = nn.Linear(128, IMAGE_LATENT_DIM, True)
    def forward(self, x):
        x = self.conv1(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)
        x = x * torch.sigmoid(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ln1(x)
        x = x * torch.sigmoid(x)
        x = self.ln2(x)
        x = x * torch.sigmoid(x)
        x = self.ln3(x)
        return x
    
class SoundExtractor(FModule):
    def __init__(self):
        super(SoundExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.ln1 = nn.Linear(2048, SOUND_LATENT_DIM, True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.ln1(x)
        return x
    
class TrajectoryExtractor(FModule):
    def __init__(self):
        super(TrajectoryExtractor, self).__init__()
        self.ln1 = nn.Linear(200, 512, True)
        self.bn1 = nn.BatchNorm1d(512)
        self.lrl1 = nn.LeakyReLU(0.01)
        self.ln2 = nn.Linear(512, 512, True)
        self.bn2 = nn.BatchNorm1d(512)
        self.lrl2 = nn.LeakyReLU(0.01)
        self.ln3 = nn.Linear(512, 512, True)
        self.bn3 = nn.BatchNorm1d(512)
        self.lrl3 = nn.LeakyReLU(0.01)
        self.ln4 = nn.Linear(512, TRAJECTORY_LATENT_DIM, True)
    def forward(self, x):
        x = self.ln1(x)
        x = self.bn1(x)
        x = self.lrl1(x)
        x = self.ln2(x)
        x = self.bn2(x)
        x = self.lrl2(x)
        x = self.ln3(x)
        x = self.bn3(x)
        x = self.lrl3(x)
        x = self.ln4(x)
        return x
    
class ImageProjector(FModule):
    def __init__(self):
        super(ImageProjector, self).__init__()
        self.ln = nn.Linear(IMAGE_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundProjector(FModule):
    def __init__(self):
        super(SoundProjector, self).__init__()
        self.ln = nn.Linear(SOUND_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class TrajectoryProjector(FModule):
    def __init__(self):
        super(TrajectoryProjector, self).__init__()
        self.ln = nn.Linear(TRAJECTORY_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundProjector(FModule):
    def __init__(self):
        super(ImageSoundProjector, self).__init__()
        self.ln = nn.Linear(IMAGE_LATENT_DIM + SOUND_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageTrajectoryProjector(FModule):
    def __init__(self):
        super(ImageTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(IMAGE_LATENT_DIM + TRAJECTORY_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundTrajectoryProjector(FModule):
    def __init__(self):
        super(SoundTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(SOUND_LATENT_DIM + TRAJECTORY_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundTrajectoryProjector(FModule):
    def __init__(self):
        super(ImageSoundTrajectoryProjector, self).__init__()
        self.ln = nn.Linear(IMAGE_LATENT_DIM + SOUND_LATENT_DIM + TRAJECTORY_LATENT_DIM, COMMON_DIM, True)
    def forward(self, x):
        return self.ln(x)

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(COMMON_DIM, 10, True)
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

        # init weight
        for name, param in self.named_parameters():
            if '.bn' in name:
                continue
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, samples, labels):
        modalities = list()
        features = list()
        for key, value in samples.items():
            features.append(self.feature_extractors[key](value))
            modalities.append(key)
        combin_key = '+'.join(modalities)
        hidden = self.projectors[combin_key](torch.cat(features, dim=1))
        outputs = self.classifier(hidden)
        loss = self.CELoss(outputs, labels)
        # import pdb; pdb.set_trace()
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()