import numpy as np
import os
from tqdm import tqdm

def read_and_save_data():
    # data root folder
    partition_dict = dict()
    partition_dict['test'] = list()
    partition_dict['dev'] = list()
    for data_type in ['train', 'test']:
        data_root_path = 'benchmark/RAW_DATA/UCIHAR'
        subject_path = os.path.join(data_root_path, data_type, 'subject_'+data_type+'.txt')
        data_path = os.path.join(data_root_path, data_type, 'Inertial Signals')
        client_id_data = np.genfromtxt(str(subject_path), dtype=int)
        
        # read labels
        labels = np.genfromtxt(str(os.path.join(data_root_path, data_type, 'y_'+data_type+'.txt')), dtype=float)-1
        # read acc
        acc_x = np.genfromtxt(str(os.path.join(data_path,'body_acc_x_'+data_type+'.txt')), dtype=float)
        acc_y = np.genfromtxt(str(os.path.join(data_path,'body_acc_y_'+data_type+'.txt')), dtype=float)
        acc_z = np.genfromtxt(str(os.path.join(data_path,'body_acc_z_'+data_type+'.txt')), dtype=float)
        # read gyro
        gyro_x = np.genfromtxt(str(os.path.join(data_path,'body_gyro_x_'+data_type+'.txt')), dtype=float)
        gyro_y = np.genfromtxt(str(os.path.join(data_path,'body_gyro_y_'+data_type+'.txt')), dtype=float)
        gyro_z = np.genfromtxt(str(os.path.join(data_path,'body_gyro_z_'+data_type+'.txt')), dtype=float)
        
        
        total_acc = np.zeros((acc_x.shape[0], acc_x.shape[1]*3))
        total_gyro = np.zeros_like(total_acc)
        
        print(f'Extract feature {data_type}')
        for data_idx in range(acc_x.shape[0]):
            acc_features, gyro_features = np.zeros([128, 3]), np.zeros([128, 3])
            # 1.1 read acc data
            acc_features[:, 0] = acc_x[data_idx, :]
            acc_features[:, 1] = acc_y[data_idx, :]
            acc_features[:, 2] = acc_z[data_idx, :]
            # 1.2 normalize acc data
            mean, std = np.mean(acc_features, axis=0), np.std(acc_features, axis=0)
            acc_features = (acc_features - mean) / (std + 1e-5)
            acc_features = acc_features.reshape(-1)
            total_acc[data_idx] = acc_features
            
            # 2.1 read gyro data
            gyro_features[:, 0] = gyro_x[data_idx, :]
            gyro_features[:, 1] = gyro_y[data_idx, :]
            gyro_features[:, 2] = gyro_z[data_idx, :]
            # 2.2 normalize gyro data
            mean, std = np.mean(gyro_features, axis=0), np.std(gyro_features, axis=0)
            gyro_features = (gyro_features - mean) / (std + 1e-5)
            gyro_features = gyro_features.reshape(-1)
            total_gyro[data_idx] = gyro_features
        
        total_x = np.concatenate((total_acc, total_gyro), axis=1).reshape(acc_x.shape[0], 2, -1)

        np.save('benchmark/RAW_DATA/UCIHAR/x_{}.npy'.format(data_type), total_x , allow_pickle=True)
        np.save('benchmark/RAW_DATA/UCIHAR/y_{}.npy'.format(data_type), labels.astype(np.float32), allow_pickle=True)
        

if __name__ == '__main__':
    read_and_save_data()