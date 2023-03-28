import os
import importlib
import utils.fflow as flw
import torch
import copy
from torch.utils.data import ConcatDataset
import math

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    projector_key = option['projector_key']
    clients = [client for client in server.clients if client.projector_key == projector_key]
    print(projector_key, [client.name for client in clients])
    used_client = copy.deepcopy(clients[0])
    used_client.train_data = ConcatDataset([client.train_data for client in clients])
    used_client.valid_data = ConcatDataset([client.valid_data for client in clients])
    used_client.datavol = len(used_client.train_data)
    used_client.batch_size = len(used_client.train_data) if option['batch_size']<0 else option['batch_size']
    used_client.batch_size = int(used_client.batch_size) if used_client.batch_size>=1 else int(len(used_client.train_data)*used_client.batch_size)   
    if option['num_steps']>0:
        used_client.num_steps = option['num_steps']
        used_client.epochs = 1.0 * used_client.num_steps/(math.ceil(len(used_client.train_data)/used_client.batch_size))
    else:
        used_client.epochs = option['num_epochs']
        used_client.num_steps = used_client.epochs * math.ceil(len(used_client.train_data) / used_client.batch_size)
    server.clients = [used_client]
    server.num_clients = 1
    server.local_data_vols = [c.datavol for c in server.clients]
    server.total_data_vol = sum(server.local_data_vols)
    server.projector_key = projector_key
    # start federated optimization
    try:
        server.run(prefix_log_filename=projector_key)
    except Exception as e:
        # log the exception that happens during training-time
        print(e)
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()