# from transformers import ViltProcessor, ViltForQuestionAnswering
# from transformers import ViltModel
import utils.fflow as flw
import torch
import wandb

def main():
    # read options
    option = flw.read_option()
    print(option)
    if option['wandb']:
        wandb.init(
            entity="aiotlab",
            project='Foundation_Multimodal_FL',
            # name="{}_CW{:.2f}_CT{:.2f}_KL{:.2f}_P{:.2f}".format(option['task'], option['contrastive_weight'], option['temperature'], option['kl_weight'], option['proportion']),
            name=option['model'],
            group=option['task'],
            # group='ptbxl_reduce_missing',
            tags=[],
            config=option
            # ,
            # resume=True
        )
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    # import pdb; pdb.set_trace()
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