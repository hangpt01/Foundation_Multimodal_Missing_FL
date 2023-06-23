from benchmark.mhd_reduce_classification.dataset import MHDReduceDataset
from benchmark.mhd_reduce_classification.model.cvae import Model
import torch
from torch.utils.data import DataLoader
from utils.fflow import setup_seed
import numpy as np
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import accuracy_score
from itertools import chain, combinations
import cv2
from matplotlib import pyplot as plt
import os

IMAGE_LATENT_DIM = 32
testset = MHDReduceDataset(train=False)
batch_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('./centralize_test/cvae.pt'))

model.to(device)
model.eval()
with torch.no_grad():
    for label in tqdm(range(10)):
        os.makedirs('./centralize_test/image/{}/gen'.format(label), exist_ok=True)
        os.makedirs('./centralize_test/image/{}/source'.format(label), exist_ok=True)
        x = testset.images[testset.labels == 0][:batch_size]
        x = (x + 1.0) / 2.0 * 255.0
        for i, _x in enumerate(x):
            plt.imshow(_x[0])
            plt.savefig('./centralize_test/image/{}/source/{}.jpg'.format(label, i))
        y = torch.ones(size=(batch_size,), dtype=torch.int64, device=device, requires_grad=False)
        z = torch.randn(size=(batch_size, IMAGE_LATENT_DIM), dtype=torch.float32, device=device, requires_grad=False)
        x = model.cvae_dict['image'].decode(z, y)
        x = (x + 1.0) / 2.0 * 255.0
        x = x.type(torch.int64).cpu().numpy()
        for i, _x in enumerate(x):
            plt.imshow(_x[0])
            plt.savefig('./centralize_test/image/{}/gen/{}.jpg'.format(label, i))