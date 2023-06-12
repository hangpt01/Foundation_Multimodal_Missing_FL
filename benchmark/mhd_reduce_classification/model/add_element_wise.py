import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from itertools import chain, combinations

# IMAGE_LATENT_DIM = 64
# SOUND_LATENT_DIM = 128
# TRAJECTORY_LATENT_DIM = 16
# IMAGE_LATENT_DIM = 32
# SOUND_LATENT_DIM = 32
# TRAJECTORY_LATENT_DIM = 32
COMMON_DIM = 32

class ImageExtractor(FModule):
    def __init__(self):
        super(ImageExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.ln1 = nn.Linear(3136, 128, True)
        self.ln2 = nn.Linear(128, 128, True)
        self.ln3 = nn.Linear(128, COMMON_DIM, True)
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
        self.ln1 = nn.Linear(2048, COMMON_DIM, True)

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
        self.ln4 = nn.Linear(512, COMMON_DIM, True)
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
        hidden = None
        for key, value in samples.items():
            feature = self.feature_extractors[key](value)
            if hidden is None:
                hidden = feature
            else:
                hidden += feature
        outputs = self.classifier(hidden)
        loss = self.CELoss(outputs, labels)
        # import pdb; pdb.set_trace()
        return loss
    
    def predict(self, samples, modalities):
        outputs_dict = dict()
        for modality in modalities:
            outputs_dict[modality] = self.classifier(
                self.feature_extractors[modality](samples[modality])
            )
        result = dict()
        for combin in chain.from_iterable(
            combinations(modalities, r + 1) for r in range(len(modalities))
        ):
            combin_key = '+'.join(combin)
            result[combin_key] = torch.stack(
                [outputs_dict[modality] for modality in combin], dim=-1
            ).sum(dim=-1).argmax(dim=-1).cpu().numpy()
        return result

if __name__ == '__main__':
    model = Model()
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()