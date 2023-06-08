import argparse
from benchmark.mhd_reduce_classification.dataset import MHDReduceDataset
from benchmark.mhd_classification.dataset import MHDDataset
from centralize_test.model import Model
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from utils.fflow import setup_seed
import numpy as np
from tqdm.auto import tqdm
import os
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
test_labels = testset.labels
train_loader = DataLoader(trainset, batch_size=option['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=option['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
# import pdb; pdb.set_trace()
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
# log_filepath = "./centralize_test/{}.csv".format(key)
# with open(log_filepath, 'w') as f:
#     f.write("train_loss,test_loss,accuracy,macro_precision,macro_recall,macro_f1\n")
# os.makedirs("./centralize_test/{}".format(key), exist_ok=True)

if option['wandb']:
    wandb.init(
        entity="aiotlab",
        project='FLMultimodal',
        name=key,
        group='mhd_centralized',
        tags=[],
        config=option
    )

for epoch in tqdm(range(option['epochs'])):
    model.train()
    for index, batch in enumerate(train_loader):
        if option['modalities']:
            x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
        else:
            x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        optimizer.zero_grad()
        loss, output = model(x, y)
        if torch.isnan(loss):
            import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        if option['lr_scheduler_type'] == 'cosine':
            scheduler.step()
    model.eval()
    train_loss = 0.0
    test_loss = 0.0
    train_predict = list()
    train_labels = list()
    test_predict = list()
    with torch.no_grad():
        for batch in train_loader:
            if option['modalities']:
                x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
            else:
                x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
            y = batch[1].to(device)
            loss, output = model(x, y)
            train_loss += loss * y.shape[0]
            train_predict.extend(torch.nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().tolist())
            train_labels.extend(y.cpu().tolist())
        for batch in test_loader:
            if option['modalities']:
                x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
            else:
                x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
            y = batch[1].to(device)
            loss, output = model(x, y)
            test_loss += loss * y.shape[0]
            test_predict.extend(torch.nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().tolist())
    if scheduler and (option['lr_scheduler_type'] != 'cosine'):
        scheduler.step()
    train_predict = np.array(train_predict)
    train_labels = np.array(train_labels)
    test_predict = np.array(test_predict)
    # import pdb; pdb.set_trace()
    # with open(log_filepath, 'a') as f:
    #     f.write("{},{},{},{},{},{}\n".format(
    #         train_loss / len(trainset),
    #         test_loss / len(testset),
    #         accuracy_score(test_labels, test_predict),
    #         precision_score(test_labels, test_predict, average='macro'),
    #         recall_score(test_labels, test_predict, average='macro'),
    #         f1_score(test_labels, test_predict, average='macro')
    #     ))
    if option['wandb']:
        wandb.log({
            'train_loss': train_loss / len(trainset),
            'train_acc': accuracy_score(train_labels, train_predict),
            'test_loss': test_loss / len(testset),
            'test_acc': accuracy_score(test_labels, test_predict)
        })
    # np.save("./centralize_test/{}/Epoch{}.npy".format(key, epoch + 1), confusion_matrix(test_labels, test_predict))