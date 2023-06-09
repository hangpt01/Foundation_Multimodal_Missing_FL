import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.fmodule import FModule

# IMAGE_LATENT_DIM = 64
# SOUND_LATENT_DIM = 128
# TRAJECTORY_LATENT_DIM = 16
IMAGE_LATENT_DIM = 32
SOUND_LATENT_DIM = 32
TRAJECTORY_LATENT_DIM = 32
# COMMON_DIM = 32

class ImageEncoder(FModule):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.ln1 = nn.Linear(3136, 128, True)
        self.ln2 = nn.Linear(128, 128, True)
        self.ln_mu = nn.Linear(128, IMAGE_LATENT_DIM, True)
        self.ln_logvar = nn.Linear(128, IMAGE_LATENT_DIM, True)
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
        mu = self.ln_mu(x)
        logvar = self.ln_logvar(x)
        return mu, logvar
    
class ImageDecoder(FModule):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self.ln1 = nn.Linear(IMAGE_LATENT_DIM, 128, True)
        self.ln2 = nn.Linear(128, 128, True)
        self.ln3 = nn.Linear(128, 3136, True)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)
    def forward(self, x):
        x = self.ln1(x)
        x = x * torch.sigmoid(x)
        x = self.ln2(x)
        x = x * torch.sigmoid(x)
        x = self.ln3(x)
        x = x * torch.sigmoid(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.deconv1(x)
        x = x * torch.sigmoid(x)
        x = self.deconv2(x)
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x
    
class SoundEncoder(FModule):
    def __init__(self):
        super(SoundEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.ln_mu = nn.Linear(2048, SOUND_LATENT_DIM, True)
        self.ln_logvar = nn.Linear(2048, SOUND_LATENT_DIM, True)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        mu = self.ln_mu(x)
        logvar = self.ln_logvar(x)
        return mu, logvar
    
class SoundDecoder(FModule):
    def __init__(self):
        super(SoundDecoder, self).__init__()
        self.ln = nn.Linear(SOUND_LATENT_DIM, 2048, True)
        self.bn1 = nn.BatchNorm1d(2048)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.bn3= nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
    def forward(self, x):
        x = self.ln(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), 256, 8, 1)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.deconv3(x)
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x
    
class TrajectoryEncoder(FModule):
    def __init__(self):
        super(TrajectoryEncoder, self).__init__()
        self.ln1 = nn.Linear(200, 512, True)
        self.bn1 = nn.BatchNorm1d(512)
        self.lrl1 = nn.LeakyReLU(0.01)
        self.ln2 = nn.Linear(512, 512, True)
        self.bn2 = nn.BatchNorm1d(512)
        self.lrl2 = nn.LeakyReLU(0.01)
        self.ln3 = nn.Linear(512, 512, True)
        self.bn3 = nn.BatchNorm1d(512)
        self.lrl3 = nn.LeakyReLU(0.01)
        self.ln_mu = nn.Linear(512, TRAJECTORY_LATENT_DIM, True)
        self.ln_logvar = nn.Linear(512, TRAJECTORY_LATENT_DIM, True)
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
        mu = self.ln_mu(x)
        logvar = self.ln_logvar(x)
        return mu, logvar

class TrajectoryDecoder(FModule):
    def __init__(self):
        super(TrajectoryDecoder, self).__init__()
        self.ln1 = nn.Linear(TRAJECTORY_LATENT_DIM, 512, True)
        self.bn1 = nn.BatchNorm1d(512)
        self.lrl1 = nn.LeakyReLU(0.01)
        self.ln2 = nn.Linear(512, 512, True)
        self.bn2 = nn.BatchNorm1d(512)
        self.lrl2 = nn.LeakyReLU(0.01)
        self.ln3 = nn.Linear(512, 512, True)
        self.bn3 = nn.BatchNorm1d(512)
        self.lrl3 = nn.LeakyReLU(0.01)
        self.ln4 = nn.Linear(512, 200, True)
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
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x

class BaseVAE(FModule):
    def __init__(self):
        super(BaseVAE, self).__init__()
        self.encoder = NotImplemented
        self.decoder = NotImplemented
    def reparameterize(self, mu, logvar):
        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()
        if torch.isnan(std.sum()) or torch.isinf(std.sum()):
            import pdb; pdb.set_trace()
        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        recon_loss = torch.sum(F.mse_loss(
            input=out.view(out.size(0), -1),
            target=x.view(x.size(0), -1),
            reduction='none'
        ), dim=-1)
        kl_div = - 0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if torch.isnan(recon_loss.mean(dim=-1)) or torch.isinf(kl_div.mean(dim=-1)):
            import pdb; pdb.set_trace()
        return {
            'recon_loss': recon_loss.mean(dim=-1),
            'kl_div': kl_div.mean(dim=-1)
        }
    
class ImageVAE(BaseVAE):
    def __init__(self):
        super(ImageVAE, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

class SoundVAE(BaseVAE):
    def __init__(self):
        super(SoundVAE, self).__init__()
        self.encoder = SoundEncoder()
        self.decoder = SoundDecoder()

class TrajectoryVAE(BaseVAE):
    def __init__(self):
        super(TrajectoryVAE, self).__init__()
        self.encoder = TrajectoryEncoder()
        self.decoder = TrajectoryDecoder()

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        # vae modules
        self.image_vae = ImageVAE()
        self.sound_vae = SoundVAE()
        self.trajectory_vae = TrajectoryVAE()
        # init weight
        for name, param in self.named_parameters():
            if 'bn' in name:
                continue
            if name.endswith('.weight'):
                nn.init.xavier_uniform_(param.data)
            elif name.endswith('.bias'):
                nn.init.zeros_(param.data)
    def forward(self, samples):
        if 'image' in samples.keys():
            image_vae_respond = self.image_vae(samples['image'])
            image_recon_loss = image_vae_respond['recon_loss']
            image_kl_div = image_vae_respond['kl_div']
        else:
            image_recon_loss = 0.0
            image_kl_div = 0.0
        if 'sound' in samples.keys():
            sound_vae_respond = self.sound_vae(samples['sound'])
            sound_recon_loss = sound_vae_respond['recon_loss']
            sound_kl_div = sound_vae_respond['kl_div']
        else:
            sound_recon_loss = 0.0
            sound_kl_div = 0.0
        if 'trajectory' in samples.keys():
            trajectory_vae_respond = self.trajectory_vae(samples['trajectory'])
            trajectory_recon_loss = trajectory_vae_respond['recon_loss']
            trajectory_kl_div = trajectory_vae_respond['kl_div']
        else:
            trajectory_recon_loss = 0.0
            trajectory_kl_div = 0.0
        return {
            'image_recon_loss': image_recon_loss,
            'image_kl_div': image_kl_div,
            'sound_recon_loss': sound_recon_loss,
            'sound_kl_div': sound_kl_div,
            'trajectory_recon_loss': trajectory_recon_loss,
            'trajectory_kl_div': trajectory_kl_div,
        }

if __name__ == '__main__':
    model = Model()
    model.eval()
    with torch.no_grad():
        result = model({
            'image': 2.0 * torch.rand(size=(64, 1, 28, 28), dtype=torch.float32, requires_grad=False) - 1.0,
            'sound': 2.0 * torch.rand(size=(64, 1, 32, 128), dtype=torch.float32, requires_grad=False) - 1.0,
            'trajectory': 2.0 * torch.rand(size=(64, 200), dtype=torch.float32, requires_grad=False) - 1.0
        })
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()