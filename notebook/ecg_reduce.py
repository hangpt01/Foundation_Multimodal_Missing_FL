import numpy as np
from ortools.sat.python import cp_model
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine

y_train = np.load('../benchmark/RAW_DATA/PTBXL/y_train.npy')
y_test = np.load('../benchmark/RAW_DATA/PTBXL/y_test.npy')
labels = np.concatenate((y_train, y_test), axis=0)
n_samples, n_labels = labels.shape

for t in tqdm(range(1, 71)):
    model = cp_model.CpModel()
    a = dict()
    for y in range(0, n_labels):
        a[y] = model.NewIntVar(0, 1, 'a[%d]' % (y))
    b = dict()
    for x in range(0, n_samples):
        b[x] = model.NewIntVar(0, 1, 'b[%d]' % (x))
        model.Add(sum([labels[x, y] * a[y] for y in range(0, n_labels)]) == 1).OnlyEnforceIf(b[x])
        model.Add(sum([labels[x, y] * a[y] for y in range(0, n_labels)]) != 1).OnlyEnforceIf(b[x].Not())
    model.Add(sum([a[y] for y in range(0, n_labels)]) >= t)
    model.Maximize(sum([b[x] for x in range(0, n_samples)]))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        s = ''
        s += 't = {}\n'.format(t)
        s += 'Keep rate = {}\n'.format(solver.ObjectiveValue() / n_samples)
        keep_ys = list()
        for y in range(0, n_labels):
            if solver.Value(a[y]) == 1:
                keep_ys.append(y)
        s += 'Keep n labels = {}\n'.format(len(keep_ys))
        tmp = list()
        for y in keep_ys:
            tmp.append(labels[:, y].sum())
        tmp = np.array(tmp)
        s += 'Balance metric = {}\n'.format(cosine(tmp, np.ones_like(tmp)))
        s += 'Count: {}\n'.format(tmp)
        s += '==============================\n\n'
        with open('./test.txt', 'a') as f:
            f.write(s)