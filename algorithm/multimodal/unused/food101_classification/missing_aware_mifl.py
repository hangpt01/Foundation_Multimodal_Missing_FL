from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
import os
import torch
import numpy as np
from transformers import ViltModel
from datetime import datetime

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.list_testing_leads = [
            [0],                    #1: Image-only
            [1],                    #2: Text-only
            [0,1],                  #3: Full
        ]
        self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        for param in self.backbone.parameters():
            param.requires_grad = False

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


    @torch.no_grad()
    def aggregate(self, models: list, modalities_list: list):
        print("Calculating clients' aggregated models ...")
        n_models = len(models)
        for k in range(n_models):
            self.clients[self.selected_clients[k]].agg_model = copy.deepcopy(self.model)
        d_q = torch.zeros(size=(n_models, n_models))
        for k in range(n_models):
            for l in range(n_models):
                d_q[k, l] = 1 + len(set(modalities_list[k]).intersection(set(modalities_list[l])))
        # modal_dict = dict()
        A = torch.zeros(size=(self.n_leads+2, n_models, n_models))
        # A = torch.zeros(size=(self.n_leads + 1, n_models, n_models))

        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.image_prompt.parameters()
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[0, k, l] = att_mat[k, l]
        
        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.text_prompt.parameters()
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[1, k, l] = att_mat[k, l]

                params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.complete_prompt.parameters()
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[2, k, l] = att_mat[k, l]
        
        # classifier
        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.classifier.parameters()
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[-1, k, l] = att_mat[k, l]

        # Assign local models       
        for k in range(n_models):
            self.clients[self.selected_clients[k]].local_model.image_prompt = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.image_prompt * \
                A[0, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ]) / sum([
                A[0, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ])
            self.clients[self.selected_clients[k]].local_model.text_prompt = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.text_prompt * \
                A[1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ]) / sum([
                A[1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ])
            self.clients[self.selected_clients[k]].local_model.complete_prompt = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.complete_prompt * \
                A[2, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ]) / sum([
                A[2, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ])

            self.clients[self.selected_clients[k]].local_model.classifier = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.classifier * \
                A[-1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ]) / sum([
                A[-1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ])
        
        new_model = copy.deepcopy(self.model)
        union_testing_leads = self.list_testing_leads[0]
        for i in range(1,len(self.list_testing_leads)):
            union_testing_leads = list(set(union_testing_leads) | set(self.list_testing_leads[i]))
        
        # set new model
        new_model.image_prompt = fmodule._model_average([
            self.clients[self.selected_clients[l]].local_model.image_prompt for l in range(n_models)
        ])
        new_model.text_prompt = fmodule._model_average([
            self.clients[self.selected_clients[l]].local_model.text_prompt for l in range(n_models)
        ])
        new_model.complete_prompt = fmodule._model_average([
            self.clients[self.selected_clients[l]].local_model.complete_prompt for l in range(n_models)
        ])

        new_model.classifier = fmodule._model_average([
            self.clients[self.selected_clients[l]].local_model.classifier for l in range(n_models)
        ])
        return new_model


    @torch.no_grad()
    # def aggregate(self, models: list, modalities_list: list):
    #     new_model = copy.deepcopy(self.model)
    #     # feature extractor
    #     for m in range(self.n_leads):
    #         p = list()
    #         chosen_models = list()
    #         for k, client_id in enumerate(self.selected_clients):
    #             if m in modalities_list[k]:
    #                 p.append(self.clients[client_id].datavol)
    #                 chosen_models.append(models[k])
    #         if len(p) == 0:
    #             continue

    #     p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        
    #     # prompt
    #     new_model.image_prompt = fmodule._model_sum([
    #         model.image_prompt * pk for model, pk in zip(models, p)
    #     ]) / sum(p)
    #     new_model.text_prompt = fmodule._model_sum([
    #         model.text_prompt * pk for model, pk in zip(models, p)
    #     ]) / sum(p)
    #     new_model.complete_prompt = fmodule._model_sum([
    #         model.complete_prompt * pk for model, pk in zip(models, p)
    #     ]) / sum(p)
        
    #     # classifier
    #     new_model.classifier = fmodule._model_sum([
    #         model.classifier * pk for model, pk in zip(models, p)
    #     ]) / sum(p)
    #     return new_model
    
    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
            "backbone": self.backbone
        }

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
                backbone=self.backbone,
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
            client_metrics = c.test(self.model, self.backbone, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 2
        self.fedmsplit_prox_lambda = option['fedmsplit_prox_lambda']
        self.modalities = modalities
        self.local_model = None
        self.agg_model = None

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        # model, backbone = self.unpack(svr_pkg)
        # self.train(model, backbone)
        # cpkg = self.pack(model)

        model, backbone = self.unpack(svr_pkg)
        if self.local_model is None:
            self.local_model = copy.deepcopy(model)
        if self.agg_model is None:
            self.agg_model = copy.deepcopy(model)
        self.train(self.local_model, backbone)
        cpkg = self.pack(self.local_model)
        return cpkg
    
    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model'], received_pkg['backbone']

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
    def train(self, model, backbone):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        for parameter in self.agg_model.parameters():
            parameter.requires_grad = False
        model.train()
        optimizer = self.calculator.get_optimizer(
            model=model,
            # backbone=backbone,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            # if batch_data[-1].shape[0] == 1:
            #     continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            
            # import pdb; pdb.set_trace()
            loss = self.calculator.train_one_step(
                model=model,
                backbone=backbone,
                data=batch_data,
                leads=self.modalities
            )['loss']
            # print(datetime.now(),iter, loss)
            regular_loss = 0.0
            if self.fedmsplit_prox_lambda > 0.0:
                for parameter, agg_parameter in zip(model.image_prompt.parameters(), self.agg_model.image_prompt.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                for parameter, agg_parameter in zip(model.text_prompt.parameters(), self.agg_model.text_prompt.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                for parameter, agg_parameter in zip(model.complete_prompt.parameters(), self.agg_model.complete_prompt.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                for parameter, agg_parameter in zip(model.classifier.parameters(), self.agg_model.classifier.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                loss += self.fedmsplit_prox_lambda * regular_loss
            loss.backward()
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, backbone, dataflag='train'):
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
            backbone=backbone,
            dataset=dataset,
            leads=self.modalities
        )