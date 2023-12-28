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
                flw.logger.log_once()
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
        return

    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        conmmunitcation_result = self.communicate(self.selected_clients)
        models = conmmunitcation_result['model']
        modalities_list = conmmunitcation_result['modalities']
        self.model = self.aggregate(models, modalities_list)
        return

    # @torch.no_grad()
    # def aggregate(self, models: list, modalities_list: list):
    #     ds = copy.deepcopy(models)
    #     delta = copy.deepcopy(self.model)

    #     n_models = len(models)
        
    #     taus = []
    #     for k in range(n_models):
    #         taus.append(self.clients[self.selected_clients[k]].num_steps)
        
    #     p = [self.clients[client_id].datavol for client_id in self.selected_clients]    
    #     p = [pk / sum(p) for pk in p]
        
    #     # create ds
    #     for m in range(self.n_leads):
    #         for k, client_id in enumerate(self.selected_clients):
    #             if m in modalities_list[k]:
    #                 ds[client_id].feature_extractors[m] = (models[client_id].feature_extractors[m] - self.model.feature_extractors[m]) / taus[client_id]
    #     for k in range(n_models):
    #         ds[k].classifier = (models[k].classifier - self.model.classifier) / taus[k]

    #     # weighted com
    #     # tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
        
    #     # for m in range(self.n_leads):
    #     #     delta.feature_extractors[m] = fmodule._model_sum([dk.feature_extractors[m] * pk for dk, pk in zip(ds, p)])
    #     # delta.classifier = fmodule._model_sum([dk.classifier * pk for dk, pk in zip(ds, p)])
        
    #     # uniform
    #     tau_eff = sum(taus) / len(taus)
    #     for m in range(self.n_leads):
    #         delta.feature_extractors[m] = fmodule._model_average([dk.feature_extractors[m] for dk in ds])
    #     delta.classifier = fmodule._model_average([dk.classifier for dk in ds])
        
        
    #     # new global model
    #     new_model = self.model + tau_eff * delta
    #     return new_model
    
    
    @torch.no_grad()
    def aggregate(self, models: list, modalities_list: list):
        new_model = copy.deepcopy(self.model)
        n_models = len(models)
        
        taus = []
        for k in range(n_models):
            taus.append(self.clients[self.selected_clients[k]].num_steps)
        
        p = [self.clients[client_id].datavol for client_id in self.selected_clients]    
        p = [pk / sum(p) for pk in p]
        
        ds_fe = []
        # create ds len (M+1)
        for m in range(self.n_leads):
            ds_element = [(models[client_id].feature_extractors[m] - self.model.feature_extractors[m]) / taus[client_id] for client_id in self.selected_clients]
            ds_fe.append(ds_element)
        ds_classifier = [(models[k].classifier - self.model.classifier) / taus[k] for k in range(self.n_leads)]

        # weighted com
        tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
        
        delta_fe = []
        for m in range(self.n_leads):
            delta_fe.append(fmodule._model_sum([dk * pk for dk, pk in zip(ds_fe[m], p)]))
        delta_classifier = fmodule._model_sum([dk * pk for dk, pk in zip(ds_classifier, p)])
        
        # # uniform
        # tau_eff = sum(taus) / len(taus)
        # for m in range(self.n_leads):
        #     delta.feature_extractors[m] = fmodule._model_average([dk.feature_extractors[m] for dk in ds])
        # delta.classifier = fmodule._model_average([dk.classifier for dk in ds])
        
        
        # new global model
        for m in range(self.n_leads):
            new_model.feature_extractors[m] = self.model.feature_extractors[m] + tau_eff * delta_fe[m]
        new_model.classifier = self.model.classifier + tau_eff * delta_classifier
        
        return new_model
    
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

    def test_on_clients(self, dataflag='train'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for client_id in self.selected_clients:
            c = self.clients[client_id]
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 2
        self.modalities = modalities

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "modalities": self.modalities
        }

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
            # import pdb; pdb.set_trace()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss, outputs = self.calculator.train_one_step(
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
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
        return self.calculator.test(
            model=model,
            dataset=dataset,
            leads=self.modalities
        )