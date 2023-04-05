import argparse
from benchmark.mhd_classification.dataset import MHDDataset
from benchmark.mhd_classification.model.mm import Model
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from utils.fflow import setup_seed
import numpy as np
from tqdm.auto import tqdm
import os

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
option = vars(parser.parse_args())
print(option)
setup_seed(option['seed'])
trainset = MHDDataset(train=True)
testset = MHDDataset(train=False)
test_labels = testset.labels.numpy()
train_loader = DataLoader(trainset, batch_size=option['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=option['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
optimizer = torch.optim.SGD(params=model.parameters(), lr=option['lr'])
criterion = torch.nn.CrossEntropyLoss()
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
log_filepath = "./centralize_test/{}.csv".format(key)
with open(log_filepath, 'w') as f:
    f.write("train_loss,test_loss,accuracy,macro_precision,macro_recall,macro_f1\n")
os.makedirs("./centralize_test/{}".format(key), exist_ok=True)

for epoch in tqdm(range(option['epochs'])):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        if option['modalities']:
            x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
        else:
            x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
        y = batch[1].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss * y.shape[0]
        if option['lr_scheduler_type'] == 'cosine':
            scheduler.step()
    model.eval()
    test_loss = 0.0
    predict = list()
    with torch.no_grad():
        for batch in test_loader:
            if option['modalities']:
                x = {modal: batch[0][modal].to(device) for modal in option['modalities']}
            else:
                x = {modal: tensor.to(device) for modal, tensor in batch[0].items()}
            y = batch[1].to(device)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss * y.shape[0]
            predict.extend(torch.nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().tolist())
    if scheduler and (option['lr_scheduler_type'] != 'cosine'):
        scheduler.step()
    predict = np.array(predict)
    with open(log_filepath, 'a') as f:
        f.write("{},{},{},{},{},{}\n".format(
            train_loss / len(trainset),
            test_loss / len(testset),
            accuracy_score(test_labels, predict),
            precision_score(test_labels, predict, average='macro'),
            recall_score(test_labels, predict, average='macro'),
            f1_score(test_labels, predict, average='macro')
        ))
    np.save("./centralize_test/{}/Epoch{}.npy".format(key, epoch + 1), confusion_matrix(test_labels, predict))