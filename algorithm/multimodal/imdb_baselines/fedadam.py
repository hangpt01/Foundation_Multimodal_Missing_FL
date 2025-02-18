from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
from tqdm import tqdm
import torch
from torch import nn
from datetime import datetime
from collections import Counter
import wandb



class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.test_data, self.other_test_datas = test_data
        # self.test_data = test_data
        self.lr = option['learning_rate']
        # self.get_missing_type_label()


    def get_missing_type_label (self):
        dataset = self.test_data
        missing_types = []
        labels = []
        for data_sample in dataset:
            missing_type = data_sample["missing_type"]
            missing_types.append(missing_type)
            label = data_sample["label"]
            labels.append(label)

        dict_types = Counter(missing_types)
        dict_labels = Counter(labels)
        print("Server")
        print({k: dict_types[k] for k in sorted(dict_types)}, '\t\t', {k: dict_labels[k] for k in sorted(dict_labels)})


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
        self.selected_clients = self.sample()
        # training
        conmmunitcation_result = self.communicate(self.selected_clients)
        models = conmmunitcation_result['model']
        self.model = self.aggregate(models)
        return

    @torch.no_grad()
    def aggregate(self, models: list):
        """
        FedAdam aggregation function.

        Args:
            self: Contains attributes like self.model, self.selected_clients, self.clients.
            models (list): List of client models after local training.

        Returns:
            new_model: Updated global model after aggregation.
        """
        # Deep copy of the global model
        new_model = copy.deepcopy(self.model)
        
        # Get data volumes of the selected clients
        p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        
        # Normalize data volumes
        total_volume = sum(p)
        p = [pk / total_volume for pk in p]
        
        # Initialize momentum and variance terms if not already defined
        if not hasattr(self, 'm'):
            self.m = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
        if not hasattr(self, 'v'):
            self.v = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
        if not hasattr(self, 't'):
            self.t = 0  # Initialize time step

        self.t += 1  # Increment time step

        # Calculate weighted average of updates
        avg_update = {name: torch.zeros_like(param) for name, param in new_model.state_dict().items()}
        for model, pk in zip(models, p):
            for name, param in model.state_dict().items():
                avg_update[name] += pk * param
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        # FedAdam updates for the global model
        global_state = new_model.state_dict()
        for name, param in global_state.items():
            # Update biased first moment
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * (avg_update[name] - param)
            
            # Update biased second moment
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * ((avg_update[name] - param) ** 2)
            
            # Compute bias-corrected moments
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            
            # Apply FedAdam update rule
            global_state[name] -= self.lr * m_hat / (torch.sqrt(v_hat) + epsilon)

        # Load the updated parameters into the global model
        new_model.load_state_dict(global_state)

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
        # print("Server in TEST", self.model)
        if self.test_data:
            result = self.calculator.server_test(
                model=model,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                option=self.option,
                current_round = self.current_round
            )
            if self.other_test_datas:
                result.update(self.calculator.server_other_test(
                    model=model,
                    datasets=self.other_test_datas,
                    batch_size=self.option['test_batch_size']
                ))
            return result
        
        else:
            return None
    
    def validate(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        if model is None: model=self.model
        if self.validation_data:
            return self.calculator.server_test(
                model=model,
                dataset=self.validation_data,
                batch_size=self.option['test_batch_size']
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
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 2
        # self.get_missing_type_label(dataflag='train')
        # self.get_missing_type_label(dataflag='valid')

    def get_missing_type_label (self, dataflag='train'):
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
        missing_types = []
        labels = []
        for data_sample in dataset:
            missing_type = data_sample["missing_type"]
            missing_types.append(missing_type)
            label = data_sample["label"]
            labels.append(label)

        dict_types = Counter(missing_types)
        dict_labels = Counter(labels)
        print(dataflag)
        print({k: dict_types[k] for k in sorted(dict_types)}, '\t\t', {k: dict_labels[k] for k in sorted(dict_labels)})


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
        # print(self.num_steps)
        # TO_DELETE
        # self.num_steps = 1
        # print(self.num_steps)

        # print("Training client", client_id+1)
        # for iter in tqdm(range(self.num_steps)):
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            # if batch_data[-1].shape[0] == 1:
            #     continue
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            
            # import pdb; pdb.set_trace()
            loss, outputs = self.calculator.train_one_step(
                model=model,
                data=batch_data
            )['loss']
            print('\t',datetime.now(),iter, loss)
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
            dataset=dataset
        )

    @fmodule.with_multi_gpus
    def test_on_specific_data(self, model, dataset, client_id, option, current_round):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        return self.calculator.test_specific_data(
            model=model,
            dataset=dataset,
            client_id=client_id,
            option=option,
            current_round = current_round
        )