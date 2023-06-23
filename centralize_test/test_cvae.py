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
confident_score_without_kl_div = dict()
if mean:
    confident_score_without_kl_div['mean'] = dict()
    for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
        combin_key = '+'.join(combin)
        confident_score_without_kl_div['mean'][combin_key] = list()
for n in mc_n_list:
    confident_score_without_kl_div['mc_{}'.format(n)] = dict()
    for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
        combin_key = '+'.join(combin)
        confident_score_without_kl_div['mc_{}'.format(n)][combin_key] = list()

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
                tmp = torch.zeros_like(nll_dict['image']['kl_div'])
                for modal in combin:
                    tmp += nll_dict[modal][method]
                prob = torch.softmax(-tmp, dim=1)
                for _prob, _y in zip(prob, y):
                    confident_score_without_kl_div[method][combin_key].append(_prob[_y].item())
with open('./centralize_test/confident_score.json', 'w') as f:
    json.dump(confident_score, f)
with open('./centralize_test/confident_score_without_kl_div.json', 'w') as f:
    json.dump(confident_score_without_kl_div, f)


# test_predict = dict()
# if mean:
#     test_predict['mean'] = dict()
#     for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
#         combin_key = '+'.join(combin)
#         test_predict['mean'][combin_key] = list()
# for n in mc_n_list:
#     test_predict['mc_{}'.format(n)] = dict()
#     for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
#         combin_key = '+'.join(combin)
#         test_predict['mc_{}'.format(n)][combin_key] = list()

# model.to(device)
# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_loader):
#         x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
#         y = batch[1].to(device)
#         nll_dict = model.predict_details(x, mean=mean, mc_n_list=mc_n_list)
#         for method in test_predict.keys():
#             for combin in chain.from_iterable(combinations(modalities, r) for r in range(1, len(modalities) + 1)):
#                 combin_key = '+'.join(combin)
#                 tmp = torch.zeros_like(nll_dict['image']['kl_div'])
#                 for modal in combin:
#                     tmp += nll_dict[modal][method] + nll_dict[modal]['kl_div']
#                 prob = torch.softmax(-tmp, dim=1)
#                 # import pdb; pdb.set_trace()
#                 test_predict[method][combin_key].extend(prob.argmax(dim=1).cpu().tolist())

# with open('./centralize_test/test_predict.json', 'w') as f:
#     json.dump(test_predict, f)
                