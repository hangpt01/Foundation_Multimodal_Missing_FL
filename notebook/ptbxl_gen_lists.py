import numpy as np
from typing import Callable, Optional
from itertools import combinations
import random
import math

def gen_list_modalities (min_num_modal, max_num_modal, NUM_USER=20):
    mat_modals = []
    list_modals_tuples = []
    for i in range(min_num_modal, max_num_modal+1):
        list_modals_tuples.append(tuple(random.sample(range(0, 12), i))) 
    for i in range(NUM_USER-(max_num_modal-min_num_modal+1)):
        gen = tuple(random.sample(range(0, 12), random.randint(min_num_modal, max_num_modal)))
        while gen in list_modals_tuples:
            gen = tuple(random.sample(range(0, 12), random.randint(min_num_modal, max_num_modal)))
            
        list_modals_tuples.append(gen)

    # print(list_modals_tuples)
    return list_modals_tuples    

def take_least_combi(num_combi_test, ls_combi_train):
    possible_coms = list(combinations(range(12),num_combi_test))
    num_exist = np.zeros(len(possible_coms))
    max_num_exist = np.zeros(len(possible_coms))
    for i in range(len(possible_coms)):
        num_exist_ls = np.zeros(len(ls_combi_train))
        for j in range(len(ls_combi_train)):
            # print(len(list(set(possible_coms[i]) & set(j))),set(possible_coms[i]),set(j))
            num_exist_ls[j] = len(list(set(possible_coms[i]) & set(ls_combi_train[j])))
            num_exist[i] += math.pow(16,num_exist_ls[j])
        # print(num_exist[i])
        max_num_exist[i] = max(num_exist_ls)
        # print(max_num_exist[i])
    sort_ind = np.argsort(num_exist)
    # print(sort_ind)
    string = possible_coms[sort_ind[0]], num_exist[sort_ind[0]], "max", max_num_exist[sort_ind[0]], possible_coms[sort_ind[1]], num_exist[sort_ind[1]], "max", max_num_exist[sort_ind[1]]
    return max_num_exist[sort_ind[0]], string
    
    
if __name__ == '__main__':
    min_ = 1
    max_ = 12
    s = '' 
    for _ in range(20):
        ls = gen_list_modalities(min_,max_, 30)
        s += 'Modalist: {}\n'.format(ls)
        for i in range(1,max_):
            max_num_exist, string = take_least_combi(i, ls)
            s += '{}'.format(i)
            s += '{}\n'.format(string)
    with open('notebook/ptbxl_missing_{}_{}.txt'.format(min_, max_), 'a') as f :
        f.write(s)  