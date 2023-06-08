import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

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
    
class Classifier(FModule):
    def __init__(self, common_dim=64):
        super(Classifier, self).__init__()
        self.common_dim = common_dim
        self.ln1 = nn.Linear(self.common_dim, 256, True)
        self.ln2 = nn.Linear(256, 128, True)
        self.ln3 = nn.Linear(128, 10, True)
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        return x
    
class ImageProjector(FModule):
    def __init__(self, common_dim=64):
        super(ImageProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(128 * 7 * 7, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundProjector(FModule):
    def __init__(self, common_dim=64):
        super(SoundProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(2048, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class TrajectoryProjector(FModule):
    def __init__(self, common_dim=64):
        super(TrajectoryProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(512, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundProjector(FModule):
    def __init__(self, common_dim=64):
        super(ImageSoundProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(128 * 7 * 7 + 2048, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageTrajectoryProjector(FModule):
    def __init__(self, common_dim=64):
        super(ImageTrajectoryProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(128 * 7 * 7 + 512, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class SoundTrajectoryProjector(FModule):
    def __init__(self, common_dim=64):
        super(SoundTrajectoryProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(2048 + 512, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class ImageSoundTrajectoryProjector(FModule):
    def __init__(self, common_dim=64):
        super(ImageSoundTrajectoryProjector, self).__init__()
        self.common_dim = common_dim
        self.ln = nn.Linear(128 * 7 * 7 + 2048 + 512, self.common_dim, True)
    def forward(self, x):
        return self.ln(x)
    
class Encoder(FModule):
    def __init__(self, common_dim=64):
        super(Encoder, self).__init__()
        self.common_dim = common_dim
        self.ln1 = nn.Linear(self.common_dim, 512, True)
        self.ln2 = nn.Linear(512, 512, True)
        self.ln3 = nn.Linear(512, self.common_dim, True)
    def forward(self, x):
        x = F.silu(self.ln1(x))
        x = F.silu(self.ln2(x))
        x = F.relu(self.ln3(x))
        return x

class Model(FModule):
    def __init__(self, common_dim=64):
        super(Model, self).__init__()
        self.common_dim = common_dim
        self.modalities = ["image", "sound", "trajectory"]

        # feature extractors
        self.feature_extractors = nn.ModuleDict({
            "image": ImageExtractor(),
            "sound": SoundExtractor(),
            "trajectory": TrajectoryExtractor()
        })

        # projectors
        self.projectors = nn.ModuleDict({
            "image": ImageProjector(common_dim=self.common_dim),
            "sound": SoundProjector(common_dim=self.common_dim),
            "trajectory": TrajectoryProjector(common_dim=self.common_dim),
            "image+sound": ImageSoundProjector(common_dim=self.common_dim),
            "image+trajectory": ImageTrajectoryProjector(common_dim=self.common_dim),
            "sound+trajectory": SoundTrajectoryProjector(common_dim=self.common_dim),
            "image+sound+trajectory": ImageSoundTrajectoryProjector(common_dim=self.common_dim)
        })

        # encoder
        self.encoder = Encoder(common_dim=self.common_dim)

        # classifier
        self.classifier = Classifier(common_dim=self.common_dim)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        keys = list()
        features = list()
        for modal in self.modalities:
            if modal in x.keys():
                keys.append(modal)
                features.append(self.feature_extractors[modal](x[modal]))
        key = "+".join(keys)
        features = torch.cat(features, dim=-1)
        features = self.projectors[key](features)
        features = self.encoder(F.relu(features))
        return F.relu(features)

if __name__ == '__main__':
    model = Model()
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()