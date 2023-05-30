from utils import fmodule
import copy
import utils.fflow as flw
import utils.system_simulator as ss
from ..fedbase import BasicServer, BasicClient
import collections
from tqdm.auto import tqdm
import os
import torch
from itertools import chain, combinations

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.contrastive_weight = option['contrastive_weight']
        self.temperature = option['temperature']
        self.all_modalities = self.model.modalities
        self.all_modal_combin_list = list()
        for combin_tuple in chain.from_iterable(combinations(self.all_modalities, r) for r in range(1, len(self.all_modalities) + 1)):
            self.all_modal_combin_list.append(list(combin_tuple))
        print(self.all_modal_combin_list)
        self.feature_extractors_cnt = dict()
        for modal in self.all_modalities:
            self.feature_extractors_cnt[modal] = {
                "n_clients": len([client for client in self.clients if modal in client.modalities]),
                "n_data": sum([client.datavol for client in self.clients if modal in client.modalities])
            }
            
        self.all_modal_combins = self.all_modalities + [self.model.combin]
        self.projectors_cnt = dict()
        for combin in self.all_modal_combins:
            self.projectors_cnt[combin] = {
                "n_clients": len([client for client in self.clients if client.modal_combin == combin]),
                "n_data": sum([client.datavol for client in self.clients if client.modal_combin == combin])
            }
        print(self.feature_extractors_cnt)
        print(self.projectors_cnt)
        self.chkpt_dir = os.path.join('fedtask', self.option['task'], 'checkpoints', flw.logger.get_output_name(suffix=''))
        if self.option['start_round']:
            self.current_round = self.option['start_round'] + 1
            print("Load weight to keep training")
            self.model.load_state_dict(torch.load(os.path.join(self.chkpt_dir, 'Round{}.pt'.format(self.option['start_round']))))
        os.makedirs(self.chkpt_dir, exist_ok=True)

    def run(self, prefix_log_filename=None):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
        # for round in range(1, 201):
            ss.clock.step()
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(self.current_round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval):
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
            self.current_round += 1
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json(prefix_log_filename=prefix_log_filename)
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
        modal_combins = conmmunitcation_result['modal_combin']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, modalities_list, modal_combins)
        # self.model = models[0]
        if self.current_round % 5 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'Round{}.pt'.format(self.current_round)))
        return

    def aggregate(self, models: list, modalities_list: list, modal_combins: list):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
        :return
            the averaged result
        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)     |other
        ==========================================================================================================================
        N/K * Σ(pk * model_k)           |1/K * Σmodel_k             |(1-Σpk) * w_old + Σ(pk * model_k)  |Σ(pk/Σpk) * model_k
        """
        if len(models) == 0: return self.model
        new_model = copy.deepcopy(self.model)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            new_model.classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)]) * N / K
            # if hasattr(new_model, 'encoder'):
            #     new_model.encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)]) * N / K
            # feature extractors
            for modal in self.all_modalities:
                if self.feature_extractors_cnt[modal]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modalities in zip(self.received_clients, models, modalities_list):
                    if modal in modalities:
                        p.append(1.0 * self.local_data_vols[cid] / self.feature_extractors_cnt[modal]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                K = len(chosen_models)
                N = self.feature_extractors_cnt[modal]["n_clients"]
                new_model.feature_extractors[modal] = fmodule._model_sum([model_k.feature_extractors[modal] * pk for model_k, pk in zip(chosen_models, p)]) * N / K
            for combin in self.all_modal_combins:
                if self.projectors_cnt[combin]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modal_combin in zip(self.received_clients, models, modal_combins):
                    if combin == modal_combin:
                        p.append(1.0 * self.local_data_vols[cid] / self.projectors_cnt[combin]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                K = len(chosen_models)
                N = self.projectors_cnt[combin]["n_clients"]
                new_model.projectors[combin] = fmodule._model_sum([model_k.projectors[combin] * pk for model_k, pk in zip(chosen_models, p)]) * N / K
                
        elif self.aggregation_option == 'uniform':
            new_model.classifier = fmodule._model_average([model_k.classifier for model_k in models])
            # if hasattr(new_model, 'encoder'):
            #     new_model.encoder = fmodule._model_average([model_k.encoder for model_k in models])
            for modal in self.all_modalities:
                if self.modal_cnt[modal]["n_clients"] == 0:
                    continue
                chosen_models = list()
                for cid, model, modalities in zip(self.received_clients, models, modalities_list):
                    if modal in modalities:
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                new_model.feature_extractors[modal] = fmodule._model_average([model_k.feature_extractors[modal] for model_k in chosen_models])
            for combin in self.all_modal_combins:
                if self.projectors_cnt[combin]["n_clients"] == 0:
                    continue
                chosen_models = list()
                for cid, model, modal_combin in zip(self.received_clients, models, modal_combins):
                    if combin == modal_combin:
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                new_model.projectors[combin] = fmodule._model_average([model_k.projectors[combin] for model_k in chosen_models])

        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            w_classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)])
            new_model.classifier = (1.0 - sum(p)) * new_model.classifier + w_classifier
            # if hasattr(new_model, 'encoder'):
            #     w_encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)])
            #     new_model.encoder = (1.0 - sum(p)) * new_model.encoder + w_encoder
            for modal in self.all_modalities:
                if self.modal_cnt[modal]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modalities in zip(self.received_clients, models, modalities_list):
                    if modal in modalities:
                        p.append(1.0 * self.local_data_vols[cid] / self.modal_cnt[modal]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                w_feature_extractor = fmodule._model_sum([model_k.feature_extractors[modal] * pk for model_k, pk in zip(chosen_models, p)])
                new_model.feature_extractors[modal] = (1.0 - sum(p)) * new_model.feature_extractors[modal] + w_feature_extractor
            for combin in self.all_modal_combins:
                if self.projectors_cnt[combin]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modal_combin in zip(self.received_clients, models, modal_combins):
                    if combin == modal_combin:
                        p.append(1.0 * self.local_data_vols[cid] / self.projectors_cnt[combin]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                w_projector = fmodule._model_sum([model_k.projectors[combin] * pk for model_k, pk in zip(chosen_models, p)])
                new_model.projectors[combin] = (1.0 - sum(p)) * new_model.projectors[combin] + w_projector

        else:
            p = [self.local_data_vols[cid] for cid in self.received_clients]
            sump = sum(p)
            p = [1.0 * pk / sump for pk in p]
            new_model.classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)])
            # if hasattr(new_model, 'encoder'):
            #     new_model.encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)])
            for modal in self.all_modalities:
                if self.modal_cnt[modal]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modalities in zip(self.received_clients, models, modalities_list):
                    if modal in modalities:
                        p.append(self.local_data_vols[cid])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                sump = sum(p)
                p = [1.0 * pk / sump for pk in p]
                new_model.feature_extractors[modal] = fmodule._model_sum([model_k.feature_extractors[modal] * pk for model_k, pk in zip(chosen_models, p)])
            for combin in self.all_modal_combins:
                if self.projectors_cnt[combin]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, modal_combin in zip(self.received_clients, models, modal_combins):
                    if combin == modal_combin:
                        p.append(self.local_data_vols[cid])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                sump = sum(p)
                p = [1.0 * pk / sump for pk in p]
                new_model.projectors[combin] = fmodule._model_sum([model_k.projectors[combin] * pk for model_k, pk in zip(chosen_models, p)])
        return new_model

    # def test(self, model=None):
    #     """
    #     Evaluate the model on the test dataset owned by the server.
    #     :param
    #         model: the model need to be evaluated
    #     :return:
    #         metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
    #     """
    #     if model is None: model=self.model
    #     if self.test_data:
    #         return self.calculator.test(model, self.test_data, self.contrastive_weight, self.temperature, batch_size=self.option['test_batch_size'])
    #     else:
    #         return None

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.custom_test(
                model,
                self.test_data,
                self.contrastive_weight,
                self.temperature,
                batch_size=self.option['test_batch_size'],
                all_modal_combin_list=self.all_modal_combin_list
            )
        else:
            return None
        
    def test_on_clients(self, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        return dict()
        # all_metrics = collections.defaultdict(list)
        # for c in self.clients:
        #     client_metrics = c.test(self.model, dataflag)
        #     for met_name, met_val in client_metrics.items():
        #         all_metrics[met_name].append(met_val)
        # return all_metrics


class Client(BasicClient):
    def __init__(self,
    option, name='', train_data=None, valid_data=None, modalities=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.modalities = modalities
        self.modal_combin = "+".join(self.modalities)
        self.contrastive_weight = option['contrastive_weight']
        self.temperature = option['temperature']

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
            "modalities": self.modalities,
            "modal_combin": self.modal_combin
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
        optimizer = self.calculator.get_optimizer(model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(model, batch_data, self.contrastive_weight, self.temperature)['loss']
            loss.backward()
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, dataflag='valid'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        return self.calculator.test(model, dataset, self.contrastive_weight, self.temperature, batch_size=self.test_batch_size)

    def get_batch_data(self):
        """
        Get the batch of data
        :return:
            a batch of data
        """
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, num_workers=self.loader_num_workers))
            batch_data = next(self.data_loader)
        # clear local DataLoader when finishing local training
        self.current_steps = (self.current_steps+1) % self.num_steps
        if self.current_steps == 0:self.data_loader = None
        batch_sample = dict()
        for modal in self.modalities:
            batch_sample[modal] = batch_data[0][modal]
        return batch_sample, batch_data[1]