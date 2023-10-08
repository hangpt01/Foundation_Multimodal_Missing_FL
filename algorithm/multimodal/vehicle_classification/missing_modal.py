from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
import os
import torch
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.list_testing_leads = [
            [0],                    #1
            [1],                    #2
            [0,1],                  #3
        ]

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            ss.clock.step()
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval) and round > 1:
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once(t=round)
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        chkpt_dir = os.path.join('fedtask', self.option['task'], 'checkpoints')
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        torch.save(self.model.state_dict(), os.path.join(chkpt_dir, 'missing_modal.pt'))
        return
    
    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.server_test(
                model=model,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                leads=self.list_testing_leads
            )
        else:
            return None


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 2
        self.modalities = modalities

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        optimizer = self.calculator.get_optimizer(
            model=model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                leads=self.modalities
            )['loss']
            loss.backward()
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, dataflag='train'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        dataset = self.train_data
        return self.calculator.test(
            model=model,
            dataset=dataset,
            leads=self.modalities
        )