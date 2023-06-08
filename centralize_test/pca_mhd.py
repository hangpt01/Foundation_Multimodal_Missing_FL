import numpy as np
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os

train = torch.load('../benchmark/RAW_DATA/MHD/mhd_train.pt')
test = torch.load('../benchmark/RAW_DATA/MHD/mhd_test.pt')

save_dir = '../benchmark/RAW_DATA/MHD_REDUCE'
os.makedirs(save_dir, exist_ok=True)

# label
np.save(os.path.join(save_dir, 'train_label.npy'), train[0].numpy())
np.save(os.path.join(save_dir, 'test_label.npy'), test[0].numpy())

# image
data = train[1].reshape(train[1].shape[0], -1) * 255.0
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.35) != 0).argmax()
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# mean = rec.mean()
# std = rec.std()
# rec = (rec - mean) / std
min = rec.min()
max = rec.max()
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(rec.shape[0], 1, 28, 28)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'train_image.npy'), rec)

data = test[1].reshape(test[1].shape[0], -1) * 255.0
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.35) != 0).argmax()
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# rec = (rec - mean) / std
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(rec.shape[0], 1, 28, 28)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'test_image.npy'), rec)

# trạectory
data = (train[2] * (train[4]['max'] - train[4]['min']) + train[4]['min']).reshape(train[2].shape[0], -1)
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
# n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.60) != 0).argmax()
n_comp = 3
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# mean = rec.mean()
# std = rec.std()
# rec = (rec - mean) / std
min = rec.min()
max = rec.max()
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(train[2].shape)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'train_trajectory.npy'), rec)

data = (test[2] * (test[4]['max'] - test[4]['min']) + test[4]['min']).reshape(test[2].shape[0], -1)
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
# n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.60) != 0).argmax()
n_comp = 3
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# rec = (rec - mean) / std
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(test[2].shape)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'test_trajectory.npy'), rec)

# sound
data = (train[3] * (train[5]['max'] - train[5]['min']) + train[5]['min']).reshape(train[3].shape[0], -1)
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
# n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.55) != 0).argmax()
n_comp = 15
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# mean = rec.mean()
# std = rec.std()
# rec = (rec - mean) / std
min = rec.min()
max = rec.max()
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(train[3].shape)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'train_sound.npy'), rec)

data = (test[3] * (test[5]['max'] - test[5]['min']) + test[5]['min']).reshape(test[3].shape[0], -1)
n_comp = data.shape[-1]
pca = PCA(n_components=n_comp)
projected = pca.fit_transform(data)
# n_comp = ((np.cumsum(pca.explained_variance_ratio_) > 0.55) != 0).argmax()
n_comp = 15
rec = np.matmul(projected[:, :n_comp], pca.components_[:n_comp])
# rec = (rec - mean) / std
rec = (rec - min) / (max - min)
rec = rec * 2.0 - 1.0
rec = rec.reshape(test[3].shape)
rec = rec.astype(np.float32)
np.save(os.path.join(save_dir, 'test_sound.npy'), rec)