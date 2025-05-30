from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
from tqdm import tqdm
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import algorithm.multimodal.food101_8_classes_fixed.vision_transformer_prompts as vit
from algorithm.multimodal.food101_8_classes_fixed.nonparametric_aggregation import *
from datetime import datetime
from collections import Counter
import wandb
import os

def remove_prefix_from_state_dict(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.num_outer_loops = option['num_outer_loops']
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

        self.test_data, self.other_test_datas = test_data
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        # self.get_missing_type()

        # Load ViLT Model
        ckpt = torch.load(self.hparams_config["load_path"], map_location="cpu")
        state_dict = ckpt["state_dict"]
        # since the pre-trained max_text_len is 40,
        # we upsample the weight of position embedding to determined max_text_len
        if self.hparams_config["max_text_len"] != 40:
            state_dict['text_embeddings.position_ids'] = torch.Tensor(range(self.hparams_config["max_text_len"])).long().view(1,-1)
            pos_emb = state_dict['text_embeddings.position_embeddings.weight']
            pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(self.hparams_config["max_text_len"],768), mode='bilinear').squeeze()
            state_dict['text_embeddings.position_embeddings.weight'] = pos_emb

        transformer_state_dict = remove_prefix_from_state_dict(state_dict, 'transformer.')
        text_embeddings_state_dict = remove_prefix_from_state_dict(state_dict, 'text_embeddings.')
        # Load the state_dicts into transformer and text_embeddings
        self.transformer.load_state_dict(transformer_state_dict, strict=True)
        self.text_embeddings.load_state_dict(text_embeddings_state_dict, strict=True)

        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False


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

    # @torch.no_grad()
    def aggregate(self, models: list):
        new_model = copy.deepcopy(self.model)
        n_models = len(models)
        for k in range(n_models):
            self.clients[self.selected_clients[k]].agg_model = copy.deepcopy(self.model)

        p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        
        # Aggregate other parts - not prompts
        # pooler
        new_model.pooler = fmodule._model_sum([
            model.pooler * pk for model, pk in zip(models, p)
        ]) / sum(p)
        
        # classifier
        new_model.classifier = fmodule._model_sum([
            model.classifier * pk for model, pk in zip(models, p)
        ]) / sum(p)
        
        # local prompt
        average_prompt = sum(pk * model.pool.prompt for pk, model in zip(p, models))  / sum(p)
        new_model.pool.prompt = nn.Parameter(average_prompt)
        
        for k in range(n_models):
            self.clients[self.selected_clients[k]].local_model.pooler = new_model.pooler
            self.clients[self.selected_clients[k]].local_model.classifier = new_model.classifier
            self.clients[self.selected_clients[k]].local_model.pool.prompt = new_model.pool.prompt
        
        temp = list()
        # Prompt aggregation
        num_prompts = new_model.global_pool.prompt.shape[0]
        union_prompts_checklist = torch.zeros(num_prompts,dtype=torch.int)
        nonzero_index = torch.nonzero(new_model.trained_prompts_checklist).flatten()
        union_prompts_checklist[nonzero_index] = 1
        # import pdb; pdb.set_trace()
        for client_idx in range(n_models):
            # import pdb; pdb.set_trace()
            nonzero_index = torch.nonzero(models[client_idx].trained_prompts_checklist).flatten()
            # print(client_idx, union_prompts_checklist, nonzero_index)
            union_prompts_checklist[nonzero_index] = 1
            # print(client_idx, union_prompts_checklist)
        for client_idx in range(n_models):
            # import pdb; pdb.set_trace()
            temp.append(models[client_idx].global_pool.prompt[torch.nonzero(union_prompts_checklist).flatten()].clone())
        
        temp = torch.stack(temp, dim=0) # temp is n_clients x prompt_length x 768: 20, 5, 768
        agg = NonparametricAgg(768, n_hidden=128).to(temp.device)
        # import pdb; pdb.set_trace()
        temp = agg(temp, outer_loop=self.num_outer_loops)
        # print("Passed one")
        #print(temp.shape)
        dataset = "food101"
        model = "L2P_Prob_only_global"
        save_file = False
        if save_file:
            if self.current_round % 25 == 0 or self.current_round == 1:
                output_dir = f"output/{dataset}/{model}/server/"
                os.makedirs(output_dir, exist_ok=True)
                file_path = f"{output_dir}summarizing_prompts_round_{self.current_round}.pt"
                torch.save(temp.detach().cpu(), file_path)

        del agg
        with torch.no_grad():
            new_model.global_pool.prompt = nn.Parameter(temp, requires_grad=True)
            for k in range(n_models):
                self.clients[self.selected_clients[k]].local_model.global_pool.prompt = new_model.global_pool.prompt
        print("Temp", temp.shape[0])
                
        new_model.reset_trained_prompts_checklist()
        for k in range(n_models):
            self.clients[self.selected_clients[k]].local_model.reset_trained_prompts_checklist()
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
            "client_id": client_id,
            "current_round": self.current_round
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
            result = self.calculator.server_test(
                model=model,
                transformer=self.transformer,
                text_embeddings=self.text_embeddings,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                option=self.option,
                current_round = self.current_round
            )
            if self.other_test_datas:
                result.update(self.calculator.server_other_test(
                    model=model,
                    transformer=self.transformer,
                    text_embeddings=self.text_embeddings,
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
                transformer=self.transformer,
                text_embeddings=self.text_embeddings,
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
        # This function uses global model after aggregation to tr
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
        self.local_model = None
        self.agg_model = None
        self.reduce_sim_scalar = option['reduce_sim_scalar']
        self.fedprox_lambda = option['fedprox_lambda']

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
        model, transformer, text_embeddings, client_id, current_round = self.unpack(svr_pkg)
        # self.client_id = client_id
        if self.local_model is None:
            self.local_model = copy.deepcopy(model)
        if self.agg_model is None:
            self.agg_model = copy.deepcopy(model)
        self.train(self.local_model, transformer, text_embeddings, client_id, current_round)
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
        return received_pkg['model'], received_pkg['transformer'], received_pkg['text_embeddings'], received_pkg['client_id'], received_pkg['current_round']


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
    def train(self, model, transformer, text_embeddings, client_id, current_round):
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
            reduce_sim, loss, outputs = self.calculator.train_one_step(
                model=model,
                transformer=transformer,
                text_embeddings=text_embeddings,
                data=batch_data, 
                client_id=client_id,
                current_round=current_round # set this to True when training the model locally, otherwise, it's False when the model is received from the server
            )['loss']
            loss = loss + self.reduce_sim_scalar * reduce_sim[0] + self.reduce_sim_scalar * reduce_sim[1]
            regular_loss = 0.0
            # import pdb; pdb.set_trace()
            if self.fedprox_lambda > 0.0:
                regular_loss += torch.sum((model.pool.prompt - self.agg_model.pool.prompt) ** 2)

                for parameter, agg_parameter in zip(model.pooler.parameters(), self.agg_model.pooler.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                for parameter, agg_parameter in zip(model.classifier.parameters(), self.agg_model.classifier.parameters()):
                    regular_loss += torch.sum(torch.pow(parameter - agg_parameter, 2))
                loss += (self.fedprox_lambda / 2) * regular_loss
            
            # import pdb; pdb.set_trace()
            # print('\t',datetime.now(),iter, loss)
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
    
    @fmodule.with_multi_gpus
    def test_on_specific_data(self, model, transformer, text_embeddings, dataset, client_id, option, current_round):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        return self.calculator.test_specific_data(
            model=model,
            transformer=transformer,
            text_embeddings=text_embeddings,
            dataset=dataset,
            client_id=client_id,
            option=option,
            current_round = current_round
        )
        
        
