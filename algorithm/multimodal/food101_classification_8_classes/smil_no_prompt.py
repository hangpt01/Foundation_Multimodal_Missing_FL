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
import algorithm.multimodal.food101_classification_arrow.vision_transformer_prompts as vit
from datetime import datetime
from collections import Counter
from sklearn.cluster import KMeans
import wandb

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 2
        self.hparams_config = {'prompt_type': 'input', 
                                'prompt_length': 16, 
                                'learnt_p': True, 
                                'prompt_layers': [0, 1, 2, 3, 4, 5], 
                                'multi_layer_prompt': True, 
                                'max_text_len': 40, 
                                'vocab_size': 30522, 
                                'vit': 'vit_base_patch32_384', 
                                'hidden_size': 768, 
                                'num_heads': 12, 
                                'num_layers': 12, 
                                'drop_rate': 0.1,
                                'mlp_ratio': 4,
                                'max_image_len': 40}
        
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
        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False

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
        # metrics_dict = dict()
        # for client_id in self.selected_clients:
        #     c = self.clients[client_id]
        #     # # import pdb; pdb.set_trace()
        #     # client_metrics = c.test(self.model, self.transformer, self.text_embeddings, dataflag)
        #     # for met_name, met_val in client_metrics.items():
        #     #     all_metrics[met_name].append(met_val)
        #     client_global_data_metrics = c.test_on_specific_data(models[client_id], self.transformer, self.text_embeddings, self.test_data, client_id, self.option, self.current_round)
        #     # loss_name = "client_" + str(client_id+1) + "_loss_global_data"
        #     # acc_name = "client_" + str(client_id+1) + "_acc_global_data"
        #     metrics_dict["client_" + str(client_id+1) + "_loss_global_data"] = (client_global_data_metrics['loss'])
        #     metrics_dict["client_" + str(client_id+1) + "_acc_global_data"] = (client_global_data_metrics['acc'])
        # if self.option['wandb']:
        #     wandb.log(metrics_dict, step=self.current_round)

        new_model = copy.deepcopy(self.model)
        p = list()
        chosen_models = list()
        for k, client_id in enumerate(self.selected_clients):
            p.append(self.clients[client_id].datavol)
            chosen_models.append(models[k])
            
        p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        
        # pooler
        new_model.pooler = fmodule._model_sum([
            model.pooler * pk for model, pk in zip(models, p)
        ]) / sum(p)
        
        # classifier
        new_model.classifier = fmodule._model_sum([
            model.classifier * pk for model, pk in zip(models, p)
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
        gpus = option['gpu']
        self.device = torch.device('cpu') if gpus is None else torch.device('cuda:{}'.format(gpus[0]))
        self.text_mean = None
        # print(device)
        # import pdb; pdb.set_trace()
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
        self.train(model, transformer, text_embeddings, client_id)
        cpkg = self.pack(model)
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

    # @ss.with_completeness
    # @fmodule.with_multi_gpus
    # def train(self, model, transformer, text_embeddings, client_id):
    #     """
    #     Standard local training procedure. Train the transmitted model with local training dataset.
    #     :param
    #         model: the global model
    #     :return
    #     """
    #     model.train()
    #     optimizer = self.calculator.get_optimizer(
    #         model=model,
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #         momentum=self.momentum
    #     )
    #     # print(self.num_steps)
    #     # TO_DELETE
    #     self.num_steps = 1
    #     # print(self.num_steps)

    #     # print("Training client", client_id+1)
    #     # for iter in tqdm(range(self.num_steps)):
    #     for iter in range(self.num_steps):
    #         # get a batch of data
    #         batch_data = self.get_batch_data()
    #         # if batch_data[-1].shape[0] == 1:
    #         #     continue
    #         model.zero_grad()
    #         # calculate the loss of the model on batched dataset through task-specified calculator
            
    #         # import pdb; pdb.set_trace()
    #         _, loss, outputs = self.calculator.train_one_step(
    #             model=model,
    #             transformer=transformer,
    #             text_embeddings=text_embeddings,
    #             data=batch_data
    #         )['loss']
    #         print('\t',datetime.now(),iter, loss)
    #         loss.backward()
    #         optimizer.step()
    
    
    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model, transformer, text_embeddings, client_id, meta_lr=1e-3, inner_steps=2, n_clusters=10):
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
        self.num_steps = 1
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            
            batch = self.data_to_device(batch_data)
            model.to(self.device)
            transformer.to(self.device)
            text_embeddings.to(self.device)
            # calculate the loss of the model on batched dataset through task-specified calculator
            cloned_model = model
            # cloned_model.load_state_dict(model.state_dict())

            # Inner loop optimization on cloned model
            import pdb; pdb.set_trace()
            # for data in batch:
            support_set, query_set = self.divide_dict(batch)
                
            inner_optimizer = torch.optim.Adam(cloned_model.parameters(), lr=meta_lr)
            # import pdb; pdb.set_trace() 
            for idx in range(batch["image"][0].shape[0]):
                for _ in range(inner_steps):
                    if batch["missing_type"][idx] == 0:
                        loss, _ , _ = cloned_model(transformer, text_embeddings, batch)
                    elif batch["missing_type"][idx] == 1:
                        loss, _ , _ = cloned_model(transformer, text_embeddings, batch, missing_text=True)
                    elif batch["missing_type"][idx] == 2:
                        loss, _ , _ = cloned_model(transformer, text_embeddings, batch, missing_image=True)

                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
            
                if batch["missing_type"][idx] == 0:
                    loss, _ , _ = cloned_model(transformer, text_embeddings, batch)
                elif batch["missing_type"][idx] == 1:
                    loss, _ , _ = cloned_model(transformer, text_embeddings, batch, missing_text=True)
                elif batch["missing_type"][idx] == 2:
                    loss, _ , _ = cloned_model(transformer, text_embeddings, batch, missing_image=True)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # KMeans clustering on the reconstructed features
            if self.text_mean == None: 
                self.text_mean = torch.rand(768, 768).to(self.device)
            reconstructed_features = cloned_model.text_reconstruction(self.text_mean)  # Example feature reconstruction
            centroids = self.kmeans_clustering(reconstructed_features.detach().cpu().numpy(), n_clusters)
            self.text_mean = torch.from_numpy(centroids).float().to(self.device)

            print(f'Outer loop loss: {loss.item()}')

            print('\t',datetime.now(),iter, loss)

    def data_to_device(self, data):
        # for k, v in data.items():
        #     print(k,len(v))
        batch = data
        # import pdb; pdb.set_trace()
        batch['image'][0] = batch['image'][0].to(self.device)
        for key in ['text_ids', 'text_labels', 'text_ids_mlm', 'text_labels_mlm', 'text_masks']:
            new_ls = []
            for tensor in data[key]:
                new_ls.append(tensor.to(self.device)) 
            batch[key] = torch.stack(new_ls)
        # batch = {k:v.to(self.device) for k,v in data.items()}
        # import pdb; pdb.set_trace()
        return batch
    

    def kmeans_clustering(self, features, n_clusters=10):
        """
        Apply KMeans clustering to the feature space.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_
        return centroids
    
    
    def divide_dict(self, input_dict, ratio=(4, 1)):
        # Calculate total number of elements
        total_elements = len(input_dict)
        
        # Calculate the number of elements for each new dictionary
        first_dict_size = (ratio[0] / sum(ratio)) * total_elements
        first_dict_size = int(round(first_dict_size))
        
        # Initialize the two dictionaries
        dict1 = {}
        dict2 = {}
        
        # Enumerate through the input dictionary and divide the elements
        for i, (key, value) in enumerate(input_dict.items()):
            if i < first_dict_size:
                dict1[key] = value
            else:
                dict2[key] = value
                
        return dict1, dict2
            
            
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