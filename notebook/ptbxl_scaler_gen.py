from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


scaler = StandardScaler()
x_train = np.load('benchmark/RAW_DATA/PTBXL_SUPERCLASS/x_train.npy', allow_pickle=True)
scaler.fit(x_train.reshape(-1, 1))

with open('benchmark/RAW_DATA/PTBXL_SUPERCLASS/standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)