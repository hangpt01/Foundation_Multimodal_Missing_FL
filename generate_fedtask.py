import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='name of the benchmark;', type=str, default='mnist_classification')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    parser.add_argument('--missing', help='missing-modality clients;', action='store_true', default=False)
    parser.add_argument('--missing_ratio_train', type=float, default=0.7)
    parser.add_argument('--missing_ratio_test', type=float, default=0.7)
    parser.add_argument('--missing_type_train', type=str, default='both')
    parser.add_argument('--missing_type_test', type=str, default='both')
    parser.add_argument('--both_ratio', type=float, default=0.5)
    parser.add_argument('--max_text_len', help='Max text len (the larger the more informative)', type=int, default=40)




    # parser.add_argument('--num_classes', help='between 3, 10, and 101 (full);', type=int, default=10)
    
    # parser.add_argument('--missing_all_6', help='same number of modalities in clients;', action='store_true', default=False)
    # parser.add_argument('--missing_1_12', help='missing with #modalities in clients (<=12);', action='store_true', default=False)
    # parser.add_argument('--missing_7_12', help='missing with #modalities in clients (6<=#modals<=12);', action='store_true', default=False)
    # parser.add_argument('--missing_rate', help='Total missing rate', type=float, default=-1)
    # parser.add_argument('--missing_ratio_2_modal', help='For 2 modality dataset: missing rate of modality 1;', type=float, default=-1)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    print(option)
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['benchmark'], 'core'])), 'TaskGen')
    generator = TaskGen(
        dist_id = option['dist'],
        skewness = option['skew'],
        num_clients=option['num_clients'],
        seed = option['seed'],
        missing = option['missing'],
        missing_ratio_train = option['missing_ratio_train'],
        missing_ratio_test = option['missing_ratio_test'],
        missing_type_train = option['missing_type_train'],
        missing_type_test = option['missing_type_test'],
        both_ratio = option['both_ratio'],
        max_text_len = option['max_text_len']
        # num_classes = option['num_classes']
        # ,
        # missing_all_6 = option['missing_all_6'],
        # missing_1_12 = option['missing_1_12'],
        # missing_7_12 = option['missing_7_12'],
        # missing_rate = option['missing_rate'],
        # missing_ratio_2_modal = option['missing_ratio_2_modal']
    )
    print(generator.taskname)
    generator.run()
