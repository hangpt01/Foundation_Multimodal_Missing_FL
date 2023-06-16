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
import json

testset = MHDReduceDataset(train=False)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('./centralize_test/cvae.pt'))
modalities = ['image', 'sound', 'trajectory']

mean = True
mc_n_list = [1, 5, 10, 50]
confident_score = dict()
if mean:
    confident_score['mean'] = dict()
    for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
        combin_key = '+'.join(combin)
        confident_score['mean'][combin_key] = list()
for n in mc_n_list:
    confident_score['mc_{}'.format(n)] = dict()
    for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
        combin_key = '+'.join(combin)
        confident_score['mc_{}'.format(n)][combin_key] = list()

model.to(device)
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        nll_dict = model.predict_details(x, mean=mean, mc_n_list=mc_n_list)
        for method in confident_score.keys():
            for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
                combin_key = '+'.join(combin)
                tmp = torch.zeros_like(nll_dict['image']['kl_div'])
                for modal in combin:
                    tmp += nll_dict[modal][method] + nll_dict[modal]['kl_div']
                prob = torch.softmax(-tmp, dim=1)
                for _prob, _y in zip(prob, y):
                    confident_score[method][combin_key].append(_prob[_y].item())
with open('./centralize_test/confident_score.json', 'w') as f:
    json.dump(confident_score, f)
                