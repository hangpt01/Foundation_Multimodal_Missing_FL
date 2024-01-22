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
    # min_ = 7
    # max_ = 12
    # s = '' 
    # for _ in range(20):
    #     ls = gen_list_modalities(min_,max_, 20)
    #     s += 'Modalist: {}\n'.format(ls)
    #     for i in range(1,max_):
    #         max_num_exist, string = take_least_combi(i, ls)
    #         s += '{}'.format(i)
    #         s += '{}\n'.format(string)
    # with open('notebook/ptbxl_reduce/ptbxl_missing_{}_{}.txt'.format(min_, max_), 'a') as f :
    #     f.write(s)  

    max_num_exist, string = take_least_combi(5, [(3, 6, 2, 7, 5, 1), (11, 2, 10, 9, 6, 1), (10, 0, 9, 6, 3, 5), (2, 1, 9, 5, 7, 6), (7, 11, 10, 8, 4, 5), (8, 9, 11, 7, 2, 10), (4, 11, 3, 0, 6, 2), (2, 0, 5, 9, 10, 11), (11, 6, 3, 9, 8, 2), (10, 9, 5, 7, 4, 11), (4, 1, 7, 2, 11, 10), (10, 1, 2, 4, 8, 9), (2, 4, 7, 0, 5, 10), (9, 10, 5, 6, 7, 1), (1, 4, 10, 8, 6, 11), (2, 9, 0, 6, 10, 1), (11, 9, 2, 5, 7, 4), (9, 4, 7, 8, 10, 5), (6, 11, 7, 1, 2, 9), (11, 5, 7, 9, 4, 2)])
    print(string)