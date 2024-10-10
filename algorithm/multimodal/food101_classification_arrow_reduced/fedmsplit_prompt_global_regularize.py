from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import algorithm.multimodal.food101_classification_arrow.vision_transformer_prompts as vit
from datetime import datetime
from collections import Counter

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.hparams_config = {'batch_size': 32, 
                        'prompt_type': 'input', 
                        'prompt_length': 16, 
                        'learnt_p': True, 
                        'prompt_layers': [0, 1, 2, 3, 4, 5], 
                        'multi_layer_prompt': True, 
                        'max_text_len': option['max_text_len'], 
                        'vocab_size': 30522, 
                        'vit': 'vit_base_patch32_384', 
                        'hidden_size': 768, 
                        'num_heads': 12, 
                        'num_layers': 12, 
                        'drop_rate': 0.1,
                        'mlp_ratio': 4,
                        'max_image_len': 40,
                        'load_path': 'benchmark/pretrained_model_weight/vilt_200k_mlm_itm.ckpt'}
        
        self.transformer = getattr(vit, self.hparams_config["vit"])(
            pretrained=False, config=self.hparams_config
        )
        bert_config = BertConfig(
            vocab_size=self.hparams_config["vocab_size"],
            hidden_size=self.hparams_config["hidden_size"],
            num_hidden_layers=self.hparams_config["num_layers"],
            num_attention_heads=self.hparams_config["num_heads"],
            intermediate_size=self.hparams_config["hidden_size"] * self.hparams_config["mlp_ratio"],
            max_position_embeddings=self.hparams_config["max_text_len"],
            hidden_dropout_prob=self.hparams_config["drop_rate"],
            attention_probs_dropout_prob=self.hparams_config["drop_rate"],
        )
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)
        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False

        # self.get_missing_type()

    def get_missing_type (self):
        dataset = self.test_data
        missing_types = []
        for data_sample in dataset:
            missing_type = data_sample["missing_type"]
            missing_types.append(missing_type)

        print(datetime.now(), "Server testing data", Counter(missing_types))

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
        self.model = self.aggregate(models)
        return


    @torch.no_grad()
    def aggregate(self, models: list):
        print("Calculating clients' aggregated models ...")
        new_model = copy.deepcopy(self.model)
        n_models = len(models)
        
        A = torch.zeros(size=(self.n_leads+3, n_models, n_models))

        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.missing_img_prompt
            ]) for k in range(n_models)
        ])
        # import pdb; pdb.set_trace()
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[0, k, l] = att_mat[k, l]
        
        # missing_text_prompt
        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.missing_text_prompt
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[1, k, l] = att_mat[k, l]

        # complete_prompt
        params = torch.stack([
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.complete_prompt
            ]) for k in range(n_models)
        ])
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)
        for k in range(n_models):
            for l in range(n_models):
                A[2, k, l] = att_mat[k, l]
        
        # pooler
        params = torch.stack([      # (20, X)
            torch.cat([
                mi.data.view(-1) for mi in \
                self.clients[self.selected_clients[k]].local_model.pooler.parameters()
            ]) for k in range(n_models)
        ])
        # import pdb; pdb.set_trace()
        dim = params.shape[1]
        att_mat = torch.softmax(params.matmul(params.T) / np.sqrt(dim), dim=1)      # (20,20)
        for k in range(n_models):
            for l in range(n_models):
                A[3, k, l] = att_mat[k, l]

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
            self.clients[self.selected_clients[k]].local_model.missing_img_prompt = nn.Parameter(sum([
                self.clients[self.selected_clients[l]].local_model.missing_img_prompt * \
                A[0, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]) / sum([
                A[0, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]))
            self.clients[self.selected_clients[k]].local_model.missing_text_prompt = nn.Parameter(sum([
                self.clients[self.selected_clients[l]].local_model.missing_text_prompt * \
                A[1, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]) / sum([
                A[1, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]))
            self.clients[self.selected_clients[k]].local_model.complete_prompt = nn.Parameter(sum([
                self.clients[self.selected_clients[l]].local_model.complete_prompt * \
                A[2, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]) / sum([
                A[2, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]))


            self.clients[self.selected_clients[k]].local_model.pooler = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.pooler * \
                A[3, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]) / sum([
                A[3, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ])

            self.clients[self.selected_clients[k]].local_model.classifier = fmodule._model_sum([
                self.clients[self.selected_clients[l]].local_model.classifier * \
                A[-1, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ]) / sum([
                A[-1, k, l] * A[:, k, l].abs().sum() for l in range(n_models)
            ])
        
        new_model = copy.deepcopy(self.model)
        # set new model
        new_model.missing_text_prompt = nn.Parameter(sum([
            self.clients[self.selected_clients[l]].local_model.missing_text_prompt for l in range(n_models)
        ]) / n_models)
        new_model.missing_img_prompt = nn.Parameter(sum([
            self.clients[self.selected_clients[l]].local_model.missing_img_prompt for l in range(n_models)
        ]) / n_models)
        new_model.complete_prompt = nn.Parameter(sum([
            self.clients[self.selected_clients[l]].local_model.complete_prompt for l in range(n_models)
        ]) / n_models)

        p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        # global model's pooler and classifier
        new_model.pooler = fmodule._model_sum([
            self.clients[self.selected_clients[l]].local_model.pooler * pk for l, pk in zip(range(n_models),p)
        ]) / sum(p)
        new_model.classifier = fmodule._model_sum([
            self.clients[self.selected_clients[l]].local_model.classifier * pk for l, pk in zip(range(n_models),p)
        ]) / sum(p)

        return new_model
    
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
            "transformer": self.transformer,
            "text_embeddings": self.text_embeddings,
            "client_id": client_id
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
                transformer=self.transformer,
                text_embeddings=self.text_embeddings,
                dataset=self.test_data,
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
            client_metrics = c.test(self.model, self.transformer, self.text_embeddings, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 2
        self.fedmsplit_prox_lambda = option['fedmsplit_prox_lambda']
        self.local_model = None
        # self.get_missing_type(dataflag='train')
        # self.get_missing_type(dataflag='valid')

    def get_missing_type (self, dataflag='train'):
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
        missing_types = []
        for data_sample in dataset:
            missing_type = data_sample["missing_type"]
            missing_types.append(missing_type)

        print(datetime.now(), dataflag, Counter(missing_types))


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
        model, transformer, text_embeddings, client_id = self.unpack(svr_pkg)
        if self.local_model is None:
            self.local_model = copy.deepcopy(model)
        self.train(self.local_model, model, transformer, text_embeddings, client_id)
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
        return received_pkg['model'], received_pkg['transformer'], received_pkg['text_embeddings'], received_pkg['client_id']


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
            "model" : model
        }

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model, regularizer_model, transformer, text_embeddings, client_id):
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
        # TO_DELETE
        # for iter in tqdm(range(self.num_steps)):
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
                transformer=transformer,
                text_embeddings=text_embeddings,
                data=batch_data
            )['loss']
            # print('\t',datetime.now(),iter, loss)
            regular_loss = 0.0
            if self.fedmsplit_prox_lambda > 0.0:
                for parameter, regularizer_parameter in zip(model.pooler.parameters(), regularizer_model.pooler.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - regularizer_parameter, 2))
                for parameter, regularizer_parameter in zip(model.classifier.parameters(), regularizer_model.classifier.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - regularizer_parameter, 2))
                loss += self.fedmsplit_prox_lambda * regular_loss

            loss.backward()
            optimizer.step()
        
        return

    @fmodule.with_multi_gpus
    def test(self, model, transformer, text_embeddings, dataflag='train'):
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
            transformer=transformer,
            text_embeddings=text_embeddings,
            dataset=dataset
        )