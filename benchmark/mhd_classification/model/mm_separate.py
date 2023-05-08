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

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(64, 10, True)
    def forward(self, x):
        return self.ln(x)

class SoundEncoder(FModule):
    def __init__(self):
        super(SoundEncoder, self).__init__()
        self.sound_extractor = SoundExtractor()
        self.projector = nn.Linear(2048, 64, True)
    def forward(self, sound):
        sound = self.sound_extractor(sound)
        return self.projector(sound)
    
class JointEncoder(FModule):
    def __init__(self):
        super(JointEncoder, self).__init__()
        self.image_extractor = ImageExtractor()
        self.sound_extractor = SoundExtractor()
        self.projector = nn.Linear(128 * 7 * 7 + 2048, 64, True)
    def forward(self, image, sound):
        image = self.image_extractor(image)
        sound = self.sound_extractor(sound)
        return self.projector(torch.concat((image, sound), dim=1))

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.modalities = ["image", "sound"]
        # encoder
        self.sound_encoder = SoundEncoder()
        self.joint_encoder = JointEncoder()

        # classifier
        self.sound_classifier = Classifier()
        self.joint_classifier = Classifier()

        # criterion
        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, samples, labels, modalities, contrastive_weight=0.0, temperature=0.0, margin=0.0, kl_weight=0.0):
        if modalities == ["image", "sound"]:
            joint_encoding = self.joint_encoder(samples["image"], samples["sound"])
            joint_outputs = self.joint_classifier(joint_encoding)
            joint_loss = self.CELoss(joint_outputs, labels)

            sound_encoding = self.sound_encoder(samples["sound"])
            sound_outputs = self.sound_classifier(sound_encoding)
            sound_loss = self.CELoss(sound_outputs, labels)

            return joint_loss, joint_outputs, sound_loss, sound_outputs
        elif modalities == ["sound"]:
            sound_encoding = self.sound_encoder(samples["sound"])
            sound_outputs = self.sound_classifier(sound_encoding)
            sound_loss = self.CELoss(sound_outputs, labels)

            return None, None, sound_loss, sound_outputs

if __name__ == '__main__':
    model = Model()
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()