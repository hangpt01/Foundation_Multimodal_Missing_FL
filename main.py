import os
import importlib
import utils.fflow as flw
import torch

def main():
    # read options
    option = flw.read_option()
    print(option)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    try:
        server.run()
    except Exception as e:
        # log the exception that happens during training-time
        print(e)
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()