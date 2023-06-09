import argparse
from benchmark.mhd_reduce_classification.dataset import MHDReduceDataset
from benchmark.mhd_reduce_classification.model.cvae import Model
# from benchmark.mhd_classification.dataset import MHDDataset
# from benchmark.mhd_classification.model.vae import Model
import torch
from torch.utils.data import DataLoader
from utils.fflow import setup_seed
import numpy as np
from tqdm.auto import tqdm
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='learning rate;', type=float)
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler type;', choices=['milestones', 'by_step', 'cosine'], default=None)
parser.add_argument('--lr_decay_rate', help='learning rate decay rate;', type=float)
parser.add_argument('--milestones', help='learning rate scheduler milestones;', nargs='+', type=int)
parser.add_argument('--step_size', help='learning rate scheduler step size;', type=int)
parser.add_argument('--batch_size', help='batch size;', type=int)
parser.add_argument('--epochs', help='number of epochs;', type=int)
parser.add_argument('--modalities', help='modalities;', nargs='+', type=str, default=[])
parser.add_argument('--seed', help='seed;', type=int, default=0)
parser.add_argument('--wandb', help='enable WANDB;', action='store_true', default=False)
option = vars(parser.parse_args())
print(option)
setup_seed(option['seed'])
trainset = MHDReduceDataset(train=True)
testset = MHDReduceDataset(train=False)
# trainset = MHDDataset(train=True)
# testset = MHDDataset(train=False)
train_loader = DataLoader(trainset, batch_size=option['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=option['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
optimizer = torch.optim.SGD(params=model.parameters(), lr=option['lr'])
model.to(device)
if option['lr_scheduler_type'] == 'milestones':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=option['milestones'], gamma=option['lr_decay_rate'])
elif option['lr_scheduler_type'] == 'by_step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=option['step_size'], gamma=option['lr_decay_rate'])
elif option['lr_scheduler_type'] == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.ceil(len(trainset) / option['batch_size']))
else:
    scheduler = None

key = "+".join(option['modalities']) if option['modalities'] else "image+sound+trajectory"

if option['wandb']:
    wandb.init(
        entity="aiotlab",
        project='FLMultimodal',
        name=key,
        group='mhd_cvae_centralized',
        tags=[],
        config=option
    )

model.eval()
train_image_recon_loss = 0.0
train_image_kl_div = 0.0
train_sound_recon_loss = 0.0
train_sound_kl_div = 0.0
train_trajectory_recon_loss = 0.0
train_trajectory_kl_div = 0.0
test_image_recon_loss = 0.0
test_image_kl_div = 0.0
test_sound_recon_loss = 0.0
test_sound_kl_div = 0.0
test_trajectory_recon_loss = 0.0
test_trajectory_kl_div = 0.0
with torch.no_grad():
    for batch in train_loader:
        if option['modalities']:
            x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
        else:
            x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        loss = model(x, y)
        train_image_recon_loss += loss['image_recon_loss'] * batch[1].shape[0]
        train_image_kl_div += loss['image_kl_div'] * batch[1].shape[0]
        train_sound_recon_loss += loss['sound_recon_loss'] * batch[1].shape[0]
        train_sound_kl_div += loss['sound_kl_div'] * batch[1].shape[0]
        train_trajectory_recon_loss += loss['trajectory_recon_loss'] * batch[1].shape[0]
        train_trajectory_kl_div += loss['trajectory_kl_div'] * batch[1].shape[0]
    for batch in test_loader:
        if option['modalities']:
            x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
        else:
            x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        loss = model(x, y)
        test_image_recon_loss += loss['image_recon_loss'] * batch[1].shape[0]
        test_image_kl_div += loss['image_kl_div'] * batch[1].shape[0]
        test_sound_recon_loss += loss['sound_recon_loss'] * batch[1].shape[0]
        test_sound_kl_div += loss['sound_kl_div'] * batch[1].shape[0]
        test_trajectory_recon_loss += loss['trajectory_recon_loss'] * batch[1].shape[0]
        test_trajectory_kl_div += loss['trajectory_kl_div'] * batch[1].shape[0]
if option['wandb']:
    wandb.log({
        'train_image_recon_loss': train_image_recon_loss / len(trainset),
        'train_image_kl_div': train_image_kl_div / len(trainset),
        'train_sound_recon_loss': train_sound_recon_loss / len(trainset),
        'train_sound_kl_div': train_sound_kl_div / len(trainset),
        'train_trajectory_recon_loss': train_trajectory_recon_loss / len(trainset),
        'train_trajectory_kl_div': train_trajectory_kl_div / len(trainset),
        'test_image_recon_loss': test_image_recon_loss / len(testset),
        'test_image_kl_div': test_image_kl_div / len(testset),
        'test_sound_recon_loss': test_sound_recon_loss / len(testset),
        'test_sound_kl_div': test_sound_kl_div / len(testset),
        'test_trajectory_recon_loss': test_trajectory_recon_loss / len(testset),
        'test_trajectory_kl_div': test_trajectory_kl_div / len(testset),
    })
for epoch in tqdm(range(option['epochs'])):
    model.train()
    for index, batch in enumerate(train_loader):
        if option['modalities']:
            x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
        else:
            x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        optimizer.zero_grad()
        loss = model(x, y)
        total_loss = sum(loss.values())
        if torch.isnan(total_loss):
            import pdb; pdb.set_trace()
        total_loss.backward()
        optimizer.step()
        if option['lr_scheduler_type'] == 'cosine':
            scheduler.step()
    model.eval()
    train_image_recon_loss = 0.0
    train_image_kl_div = 0.0
    train_sound_recon_loss = 0.0
    train_sound_kl_div = 0.0
    train_trajectory_recon_loss = 0.0
    train_trajectory_kl_div = 0.0
    test_image_recon_loss = 0.0
    test_image_kl_div = 0.0
    test_sound_recon_loss = 0.0
    test_sound_kl_div = 0.0
    test_trajectory_recon_loss = 0.0
    test_trajectory_kl_div = 0.0
    with torch.no_grad():
        for batch in train_loader:
            if option['modalities']:
                x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
            else:
                x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
            y = batch[1].to(device)
            loss = model(x, y)
            train_image_recon_loss += loss['image_recon_loss'] * batch[1].shape[0]
            train_image_kl_div += loss['image_kl_div'] * batch[1].shape[0]
            train_sound_recon_loss += loss['sound_recon_loss'] * batch[1].shape[0]
            train_sound_kl_div += loss['sound_kl_div'] * batch[1].shape[0]
            train_trajectory_recon_loss += loss['trajectory_recon_loss'] * batch[1].shape[0]
            train_trajectory_kl_div += loss['trajectory_kl_div'] * batch[1].shape[0]
        for batch in test_loader:
            if option['modalities']:
                x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
            else:
                x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
            y = batch[1].to(device)
            loss = model(x, y)
            test_image_recon_loss += loss['image_recon_loss'] * batch[1].shape[0]
            test_image_kl_div += loss['image_kl_div'] * batch[1].shape[0]
            test_sound_recon_loss += loss['sound_recon_loss'] * batch[1].shape[0]
            test_sound_kl_div += loss['sound_kl_div'] * batch[1].shape[0]
            test_trajectory_recon_loss += loss['trajectory_recon_loss'] * batch[1].shape[0]
            test_trajectory_kl_div += loss['trajectory_kl_div'] * batch[1].shape[0]
    if option['wandb']:
        wandb.log({
            'train_image_recon_loss': train_image_recon_loss / len(trainset),
            'train_image_kl_div': train_image_kl_div / len(trainset),
            'train_sound_recon_loss': train_sound_recon_loss / len(trainset),
            'train_sound_kl_div': train_sound_kl_div / len(trainset),
            'train_trajectory_recon_loss': train_trajectory_recon_loss / len(trainset),
            'train_trajectory_kl_div': train_trajectory_kl_div / len(trainset),
            'test_image_recon_loss': test_image_recon_loss / len(testset),
            'test_image_kl_div': test_image_kl_div / len(testset),
            'test_sound_recon_loss': test_sound_recon_loss / len(testset),
            'test_sound_kl_div': test_sound_kl_div / len(testset),
            'test_trajectory_recon_loss': test_trajectory_recon_loss / len(testset),
            'test_trajectory_kl_div': test_trajectory_kl_div / len(testset),
        })
    if scheduler and (option['lr_scheduler_type'] != 'cosine'):
        scheduler.step()