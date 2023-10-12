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
        # self.z_M = [1/12]*12

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
        modal_dict = dict()
        A = torch.zeros(size=(self.n_leads *3+1, n_models, n_models))
        # feature extractors update coefficients
        for m in range(self.n_leads):
            modal_dict[m] = list()
            for k in range(n_models):
                if m in modalities_list[k]:
                    modal_dict[m].append(k)
            if len(modal_dict[m]) == 0:
                continue
            params = torch.stack([
                torch.cat([
                    mi.data.view(-1) for mi in \
                    self.clients[self.selected_clients[k]].local_model.feature_extractors[m].parameters()
                ]) for k in modal_dict[m]
            ])
            dim = params.shape[1]
            att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
            for idk, k in enumerate(modal_dict[m]):
                for idl, l in enumerate(modal_dict[m]):
                    A[m, k, l] = att_mat[idk, idl]
        
        
        # relation embedders update coefficient
        list_full_n_leads = [*range(self.n_leads)]
        for m in range(self.n_leads):
            params = torch.stack([
                torch.cat([
                    mi.data.view(-1) for mi in \
                    self.clients[self.selected_clients[k]].local_model.relation_embedders[m].parameters()
                ]) for k in range(n_models)
            ])
            dim = params.shape[1]
            att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
            for k in range(n_models):
                for l in range(n_models):
                    A[self.n_leads+m, k, l] = att_mat[k, l]
        
        # classifiers update coefficient
        for m in range(self.n_leads):
            params = torch.stack([
                torch.cat([
                    mi.data.view(-1) for mi in \
                    self.clients[self.selected_clients[k]].local_model.classifiers[m].parameters()
                ]) for k in range(n_models)
            ])
            dim = params.shape[1]
            att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
            for k in range(n_models):
                for l in range(n_models):
                    A[self.n_leads*2+m, k, l] = att_mat[k, l]
        
        # multi classifier update coefficient
        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.multi_classifier.parameters()
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[-1, k, l] = att_mat[k, l]
        
        # global_relation_embedders = []
        # global_classifiers = []
        
        for m in range(self.n_leads):       #clients new feature extractors
            for k in modal_dict[m]:
                self.clients[self.selected_clients[k]].local_model.feature_extractors[m] = fmodule._model_sum([
                    self.clients[self.selected_clients[l]].local_model.feature_extractors[m] * \
                    A[m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in modal_dict[m]
                ]) / sum([
                    A[m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in modal_dict[m]
                ])
        for m in range(self.n_leads):
            for k in range(n_models):
                self.clients[self.selected_clients[k]].local_model.relation_embedders[m] = fmodule._model_sum([
                        self.clients[self.selected_clients[l]].local_model.relation_embedders[m] * \
                        A[self.n_leads+m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
                    ]) / sum([
                        A[self.n_leads+m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
                    ])
                self.clients[self.selected_clients[k]].local_model.classifiers[m] = fmodule._model_sum([
                        self.clients[self.selected_clients[l]].local_model.classifiers[m] * \
                        A[self.n_leads*2+m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
                    ]) / sum([
                        A[self.n_leads*2+m, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
                    ])
        
        for k in range(n_models):
            self.clients[self.selected_clients[k]].local_model.multi_classifier = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.multi_classifier * \
                A[-1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ]) / sum([
                A[-1, k, l] * A[:, k, l].abs().sum() / d_q[k, l] for l in range(n_models)
            ])    
            
        
        new_model = copy.deepcopy(self.model)
        union_testing_leads = self.list_testing_leads[0]
        for i in range(1,len(self.list_testing_leads)):
            union_testing_leads = list(set(union_testing_leads) | set(self.list_testing_leads[i]))
        for m in union_testing_leads:
            new_model.feature_extractors[m] = fmodule._model_average([
                self.clients[self.selected_clients[l]].local_model.feature_extractors[m] for l in modal_dict[m]
            ])
        for m in list_full_n_leads:
            new_model.relation_embedders[m] = fmodule._model_average([
                self.clients[self.selected_clients[l]].local_model.relation_embedders[m] for l in range(n_models)
            ])
            new_model.classifiers[m] = fmodule._model_average([
                self.clients[self.selected_clients[l]].local_model.classifiers[m] for l in range(n_models)
            ])
            
        new_model.multi_classifier = fmodule._model_average([
            self.clients[self.selected_clients[l]].local_model.multi_classifier for l in range(n_models)
        ])
        
        clients_z_M = [self.clients[self.selected_clients[l]].local_model.z_M for l in range(n_models)]
        new_model.z_M = torch.FloatTensor([sum(sub_list) / len(sub_list) for sub_list in zip(*clients_z_M)])
        # p_k = 
    
        print("Z_M", new_model.z_M)
        for k in range(n_models):    
            self.clients[self.selected_clients[k]].local_model.z_M = new_model.z_M
            
            #newly added
        #     self.clients[self.selected_clients[k]].local_model.p_k = self.clients[self.selected_clients[k]].local_model.delta_G / (self.clients[self.selected_clients[k]].local_model.delta_G**2 *2)
            
        # M = sum([self.clients[self.selected_clients[k]].local_model.p_k for k in range(n_models)])
        # for k in range(n_models):    
        #     self.clients[self.selected_clients[k]].local_model.p_k = self.clients[self.selected_clients[k]].local_model.p_k / M
            
            
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
        # import pdb; pdb.set_trace()
        
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
            client_metrics = c.test(c.local_model, dataflag)
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
        model = self.unpack(svr_pkg)
        if self.local_model is None:
            self.local_model = copy.deepcopy(model)
        if self.agg_model is None:
            self.agg_model = copy.deepcopy(model)
        self.train(self.local_model)
        cpkg = self.pack(self.local_model)
        return cpkg

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):     #Client training
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
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        # first step - step 0
        loss = 0
        val_loss_step_0 = torch.stack(self.evaluate(model))
        # import pdb; pdb.set_trace()
        z_M_step_0 = model.z_M.to(self.device)
        for iter in range(self.num_steps):      # num clients epochs
            # get a batch of data
            # import pdb; pdb.set_trace()
            
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            # import pdb; pdb.set_trace()
            loss, loss_ , outputs = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                leads=self.modalities
            )['loss']

            # regular_loss = 0.0
            # if self.fedmsplit_prox_lambda > 0.0:
            #     for m in self.modalities:
            #         for parameter, agg_parameter in zip(model.feature_extractors[m].parameters(), self.agg_model.feature_extractors[m].parameters()):
            #             regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
            #         for parameter, agg_parameter in zip(model.relation_embedders[m].parameters(), self.agg_model.relation_embedders[m].parameters()):
            #             regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
            #         for parameter, agg_parameter in zip(model.classifiers[m].parameters(), self.agg_model.classifiers[m].parameters()):
            #             regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))    
            #     # for parameter, agg_parameter in zip(model.classifier.parameters(), self.agg_model.classifier.parameters()):
            #     #     regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
            #     loss += self.fedmsplit_prox_lambda * regular_loss
            
            if iter==0:
                loss_step_0 = torch.stack(loss)
            # import pdb; pdb.set_trace()
            # loss.backward(retain_graph=True)
            loss_.backward()
            optimizer.step()
        
        # Training loss per batch
        loss_step_E = torch.stack(loss)
        # import pdb; pdb.set_trace()
        
        # for m in self.modalities:
        val_loss_step_E = torch.stack(self.evaluate(model))
        # import pdb; pdb.set_trace()
        
        delta_G = abs(val_loss_step_E - val_loss_step_0)
        delta_O = abs((val_loss_step_E - val_loss_step_0) - (loss_step_E - loss_step_0))
        z_M = delta_G / (delta_O*delta_O)
        Q = torch.sum(z_M)
        model.z_M = z_M / (2*Q)
        
        z_M_step_E = model.z_M
        # import pdb; pdb.set_trace()
        
        loss_total_step_0 = torch.sum(z_M_step_0*loss_step_0).item()
        loss_total_step_E = torch.sum(z_M_step_E*loss_step_E).item()
        val_loss_total_step_0 = torch.sum(z_M_step_0*val_loss_step_0).item()
        val_loss_total_step_E = torch.sum(z_M_step_0*val_loss_step_E).item()
        
        delta_G_client = abs(val_loss_total_step_E - val_loss_total_step_0)
        delta_O_client = abs((val_loss_total_step_E - val_loss_total_step_0) - (loss_total_step_E - loss_total_step_0))
        model.delta_G = delta_G_client
        model.delta_O = delta_O_client
        
        # import pdb; pdb.set_trace()
                
        return

    @fmodule.with_multi_gpus
    def evaluate(self, model):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        dataset = self.valid_data
        # import pdb; pdb.set_trace()
        return self.calculator.evaluate(
            model=model,
            dataset=dataset,
            leads=self.modalities
        )
        
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