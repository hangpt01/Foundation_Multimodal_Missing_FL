import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.fmodule import FModule

class BaseCVAE(FModule):
    def __init__(self):
        super(BaseCVAE, self).__init__()
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
    def encode(self, x, y):
        raise NotImplementedError("Subclasses should implement this!")
    def decode(self, x, y):
        raise NotImplementedError("Subclasses should implement this!")
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, y)
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

class ImageCVAE(BaseCVAE):
    def __init__(self):
        super(ImageCVAE, self).__init__()
        # encoder
        self.encoder_label_embed = nn.Linear(10, 28 * 28, bias=True)
        self.encoder_conv0 = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.encoder_conv1 = nn.Conv2d(1 + 1, 32, 4, 2, 1, bias=False)
        self.encoder_conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.encoder_ln1 = nn.Linear(3136, 128, True)
        self.encoder_ln2 = nn.Linear(128, 128, True)
        self.encoder_ln_mu = nn.Linear(128, 64, True)
        self.encoder_ln_logvar = nn.Linear(128, 64, True)
        # decoder
        self.decoder_ln1 = nn.Linear(64 + 10, 128, True)
        self.decoder_ln2 = nn.Linear(128, 128, True)
        self.decoder_ln3 = nn.Linear(128, 3136, True)
        self.decoder_deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.decoder_deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)
    def encode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        y = self.encoder_label_embed(y)
        y = y.view(y.size(0), 1, 28, 28)
        x = self.encoder_conv0(x)
        x = torch.cat((x, y), dim=1)
        x = self.encoder_conv1(x)
        x = x * torch.sigmoid(x)
        x = self.encoder_conv2(x)
        x = x * torch.sigmoid(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_ln1(x)
        x = x * torch.sigmoid(x)
        x = self.encoder_ln2(x)
        x = x * torch.sigmoid(x)
        mu = self.encoder_ln_mu(x)
        logvar = self.encoder_ln_logvar(x)
        return mu, logvar
    def decode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        x = torch.cat((x, y), dim=1)
        x = self.decoder_ln1(x)
        x = x * torch.sigmoid(x)
        x = self.decoder_ln2(x)
        x = x * torch.sigmoid(x)
        x = self.decoder_ln3(x)
        x = x * torch.sigmoid(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder_deconv1(x)
        x = x * torch.sigmoid(x)
        x = self.decoder_deconv2(x)
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x

class SoundCVAE(BaseCVAE):
    def __init__(self):
        super(SoundCVAE, self).__init__()
        # encoder
        self.encoder_label_embed = nn.Linear(10, 32 * 128, bias=True)
        self.encoder_conv0 = nn.Conv2d(1, 1, 1, 1, 0, bias=True)
        self.encoder_conv1 = nn.Conv2d(in_channels=(1 + 1), out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
        self.encoder_bn1 = nn.BatchNorm2d(128)
        self.encoder_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.encoder_bn2 = nn.BatchNorm2d(128)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.encoder_bn3 = nn.BatchNorm2d(256)
        self.encoder_ln_mu = nn.Linear(2048, 128, True)
        self.encoder_ln_logvar = nn.Linear(2048, 128, True)
        # decoder
        self.decoder_ln = nn.Linear(128 + 10, 2048, True)
        self.decoder_bn1 = nn.BatchNorm1d(2048)
        self.decoder_deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(128)
        self.decoder_deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False)
        self.decoder_bn3= nn.BatchNorm2d(128)
        self.decoder_deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False)
    def encode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        y = self.encoder_label_embed(y)
        y = y.view(y.size(0), 1, 32, 128)
        x = self.encoder_conv0(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = F.relu(self.encoder_bn3(self.encoder_conv3(x)))
        x = torch.flatten(x, start_dim=1)
        mu = self.encoder_ln_mu(x)
        logvar = self.encoder_ln_logvar(x)
        return mu, logvar
    def decode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        x = torch.cat((x, y), dim=1)
        x = self.decoder_ln(x)
        x = self.decoder_bn1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), 256, 8, 1)
        x = self.decoder_deconv1(x)
        x = self.decoder_bn2(x)
        x = torch.relu(x)
        x = self.decoder_deconv2(x)
        x = self.decoder_bn3(x)
        x = torch.relu(x)
        x = self.decoder_deconv3(x)
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x
    
class TrajectoryCVAE(BaseCVAE):
    def __init__(self):
        super(TrajectoryCVAE, self).__init__()
        # encoder
        self.encoder_label_embed = nn.Linear(10, 200, bias=True)
        self.encoder_conv0 = nn.Conv1d(1, 1, 1, 1, 0, bias=True)
        self.encoder_ln1 = nn.Linear(200 + 200, 512, True)
        self.encoder_bn1 = nn.BatchNorm1d(512)
        self.encoder_lrl1 = nn.LeakyReLU(0.01)
        self.encoder_ln2 = nn.Linear(512, 512, True)
        self.encoder_bn2 = nn.BatchNorm1d(512)
        self.encoder_lrl2 = nn.LeakyReLU(0.01)
        self.encoder_ln3 = nn.Linear(512, 512, True)
        self.encoder_bn3 = nn.BatchNorm1d(512)
        self.encoder_lrl3 = nn.LeakyReLU(0.01)
        self.encoder_ln_mu = nn.Linear(512, 16, True)
        self.encoder_ln_logvar = nn.Linear(512, 16, True)
        # decoder
        self.decoder_ln1 = nn.Linear(16 + 10, 512, True)
        self.decoder_bn1 = nn.BatchNorm1d(512)
        self.decoder_lrl1 = nn.LeakyReLU(0.01)
        self.decoder_ln2 = nn.Linear(512, 512, True)
        self.decoder_bn2 = nn.BatchNorm1d(512)
        self.decoder_lrl2 = nn.LeakyReLU(0.01)
        self.decoder_ln3 = nn.Linear(512, 512, True)
        self.decoder_bn3 = nn.BatchNorm1d(512)
        self.decoder_lrl3 = nn.LeakyReLU(0.01)
        self.decoder_ln4 = nn.Linear(512, 200, True)
    def encode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        y = self.encoder_label_embed(y)
        x = x.view(x.size(0), 1, 200)
        x = self.encoder_conv0(x)
        x = x.view(x.size(0), 200)
        x = torch.cat((x, y), dim=1)
        x = self.encoder_ln1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_lrl1(x)
        x = self.encoder_ln2(x)
        x = self.encoder_bn2(x)
        x = self.encoder_lrl2(x)
        x = self.encoder_ln3(x)
        x = self.encoder_bn3(x)
        x = self.encoder_lrl3(x)
        mu = self.encoder_ln_mu(x)
        logvar = self.encoder_ln_logvar(x)
        return mu, logvar
    def decode(self, x, y):
        y = F.one_hot(y, num_classes=10).type(torch.float32)
        x = torch.cat((x, y), dim=1)
        x = self.decoder_ln1(x)
        x = self.decoder_bn1(x)
        x = self.decoder_lrl1(x)
        x = self.decoder_ln2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_lrl2(x)
        x = self.decoder_ln3(x)
        x = self.decoder_bn3(x)
        x = self.decoder_lrl3(x)
        x = self.decoder_ln4(x)
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.image_cvae = ImageCVAE()
        self.sound_cvae = SoundCVAE()
        self.trajectory_cvae = TrajectoryCVAE()
    def forward(self, samples, labels):
        if 'image' in samples.keys():
            image_cvae_respond = self.image_cvae(samples['image'], labels)
            image_recon_loss = image_cvae_respond['recon_loss']
            image_kl_div = image_cvae_respond['kl_div']
        else:
            image_recon_loss = 0.0
            image_kl_div = 0.0
        if 'sound' in samples.keys():
            sound_cvae_respond = self.sound_cvae(samples['sound'], labels)
            sound_recon_loss = sound_cvae_respond['recon_loss']
            sound_kl_div = sound_cvae_respond['kl_div']
        else:
            sound_recon_loss = 0.0
            sound_kl_div = 0.0
        if 'trajectory' in samples.keys():
            trajectory_cvae_respond = self.trajectory_cvae(samples['trajectory'], labels)
            trajectory_recon_loss = trajectory_cvae_respond['recon_loss']
            trajectory_kl_div = trajectory_cvae_respond['kl_div']
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
        result = model(
            {
            'image': 2.0 * torch.rand(size=(64, 1, 28, 28), dtype=torch.float32, requires_grad=False) - 1.0,
            'sound': 2.0 * torch.rand(size=(64, 1, 32, 128), dtype=torch.float32, requires_grad=False) - 1.0,
            'trajectory': 2.0 * torch.rand(size=(64, 200), dtype=torch.float32, requires_grad=False) - 1.0
            },
            torch.randint(low=0, high=10, size=(64,), dtype=torch.int64,requires_grad=False)
        )
    print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()