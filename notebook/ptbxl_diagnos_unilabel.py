import pandas as pd
import numpy as np
import wfdb
import ast
import os

def load_raw_data(df, sampling_rate, path):
    # files = os.listdir(path+'records100/00000/')
    # files_unique_name = list(set([f[:-4] for f in files]))
    data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'benchmark/RAW_DATA/PTBXL_RAW/physionet.org/files/ptb-xl/1.0.3/'

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
# Y = Y.iloc[:366]
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate=100, path=path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

print("Done loading")

# Get kept indices and count y's superclasses
def get_kept_indices (y_ls):
    dict_y_all = dict()
    dict_y_kept = dict()
    keep_x_indices = []
    for i, y in enumerate(y_ls):
        if len(y) > 0:
            if len(y)==1:
                if y[0] in dict_y_kept.keys():
                    dict_y_kept[y[0]] += 1
                else:
                    dict_y_kept[y[0]] = 1
                keep_x_indices.append(i)
            else:
                y = '+'.join(y)
                if y in dict_y_all.keys():
                    dict_y_all[y] += 1
                else:
                    dict_y_all[y] = 1
    return keep_x_indices, dict_y_kept, dict_y_all


def get_kept_x_y (keep_x_indices, x_np, y_ls):
    keep_x = []
    keep_y = []
    print(len(keep_x_indices), x_np.shape, len(y_ls))
    for i in keep_x_indices:
        keep_x.append(x_np[i,:,:])
        keep_y.append(y_ls[i][0])
    keep_y = np.array(keep_y)
    keep_y[keep_y=='NORM'] = 0
    keep_y[keep_y=='CD'] = 1
    keep_y[keep_y=='MI'] = 2
    keep_y[keep_y=='HYP'] = 3
    keep_y[keep_y=='STTC'] = 4
    keep_y = keep_y.astype(int)
    return np.array(keep_x), keep_y


# Kept indices
ls_y_train = list(y_train)
ls_y_test = list(y_test)
train_indices, train_dict_y_kept, train_dict_y_all = get_kept_indices(ls_y_train)
test_indices, test_dict_y_kept, test_dict_y_all = get_kept_indices(ls_y_test)
print("Got indices")


# After choosing elements and changing labels to numerical
x_train, y_train = get_kept_x_y(train_indices, X_train, ls_y_train)
x_test, y_test = get_kept_x_y(test_indices, X_test, ls_y_test)
print("Got kept x and y")


np.save('benchmark/RAW_DATA/PTBXL_SUPERCLASS/x_train.npy', np.array(x_train, dtype=object), allow_pickle=True)
np.save('benchmark/RAW_DATA/PTBXL_SUPERCLASS/x_test.npy', np.array(x_test, dtype=object), allow_pickle=True)
np.save('benchmark/RAW_DATA/PTBXL_SUPERCLASS/y_train.npy', np.array(y_train, dtype=object), allow_pickle=True)
np.save('benchmark/RAW_DATA/PTBXL_SUPERCLASS/y_test.npy', np.array(y_test, dtype=object), allow_pickle=True)


import pdb; pdb.set_trace()
