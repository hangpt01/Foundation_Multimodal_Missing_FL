from utils import fmodule
import copy
import utils.fflow as flw
import utils.system_simulator as ss
from .fedbase import BasicServer, BasicClient

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.all_modalities = self.model.modalities
        self.modal_cnt = dict()
        for modal in self.all_modalities:
            self.modal_cnt[modal] = {
                "n_clients": len([client for client in self.clients if modal in client.modalities]),
                "n_data": sum([client.datavol for client in self.clients if modal in client.modalities])
            }
        self.projector_keys = list(self.model.projectors.keys())
        self.prj_cnt = dict()
        for key in self.projector_keys:
            self.prj_cnt[key] = {
                "n_clients": len([client for client in self.clients if client.projector_key == key]),
                "n_data": sum([client.datavol for client in self.clients if client.projector_key == key])
            }
        self.projector_key = None

    def run(self, prefix_log_filename=None):
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
        projector_keys = conmmunitcation_result['projector_key']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, modalities_list, projector_keys)
        return

    def aggregate(self, models: list, modalities_list: list, projector_keys: list):
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
            if hasattr(new_model, 'encoder'):
                new_model.encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)]) * N / K
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
                K = len(chosen_models)
                N = self.modal_cnt[modal]["n_clients"]
                new_model.feature_extractors[modal] = fmodule._model_sum([model_k.feature_extractors[modal] * pk for model_k, pk in zip(chosen_models, p)]) * N / K
            for key in self.projector_keys:
                if self.prj_cnt[key]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, projector_key in zip(self.received_clients, models, projector_keys):
                    if key == projector_key:
                        p.append(1.0 * self.local_data_vols[cid] / self.prj_cnt[key]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                K = len(chosen_models)
                N = self.prj_cnt[key]["n_clients"]
                new_model.projectors[key] = fmodule._model_sum([model_k.projectors[key] * pk for model_k, pk in zip(chosen_models, p)]) * N / K
                
        elif self.aggregation_option == 'uniform':
            new_model.classifier = fmodule._model_average([model_k.classifier for model_k in models])
            if hasattr(new_model, 'encoder'):
                new_model.encoder = fmodule._model_average([model_k.encoder for model_k in models])
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
            for key in self.projector_keys:
                if self.prj_cnt[key]["n_clients"] == 0:
                    continue
                chosen_models = list()
                for cid, model, projector_key in zip(self.received_clients, models, projector_keys):
                    if key == projector_key:
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                new_model.projectors[key] = fmodule._model_average([model_k.projectors[key] for model_k in chosen_models])

        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            w_classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)])
            new_model.classifier = (1.0 - sum(p)) * new_model.classifier + w_classifier
            if hasattr(new_model, 'encoder'):
                w_encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)])
                new_model.encoder = (1.0 - sum(p)) * new_model.encoder + w_encoder
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
            for key in self.projector_keys:
                if self.prj_cnt[key]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, projector_key in zip(self.received_clients, models, projector_keys):
                    if key == projector_key:
                        p.append(1.0 * self.local_data_vols[cid] / self.prj_cnt[key]["n_data"])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                w_projector = fmodule._model_sum([model_k.projectors[key] * pk for model_k, pk in zip(chosen_models, p)])
                new_model.projectors[key] = (1.0 - sum(p)) * new_model.projectors[key] + w_projector

        else:
            p = [self.local_data_vols[cid] for cid in self.received_clients]
            sump = sum(p)
            p = [1.0 * pk / sump for pk in p]
            new_model.classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)])
            if hasattr(new_model, 'encoder'):
                new_model.encoder = fmodule._model_sum([model_k.encoder * pk for model_k, pk in zip(models, p)])
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
            for key in self.projector_keys:
                if self.prj_cnt[key]["n_clients"] == 0:
                    continue
                p = list()
                chosen_models = list()
                for cid, model, projector_key in zip(self.received_clients, models, projector_keys):
                    if key == projector_key:
                        p.append(self.local_data_vols[cid])
                        chosen_models.append(model)
                if len(chosen_models) == 0:
                    continue
                sump = sum(p)
                p = [1.0 * pk / sump for pk in p]
                new_model.projectors[key] = fmodule._model_sum([model_k.projectors[key] * pk for model_k, pk in zip(chosen_models, p)])
        return new_model

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
                batch_size=self.option['test_batch_size'],
                modal_combin=self.projector_keys,
                save_dir="fedtask/{}/details/{}".format(self.option["task"], flw.logger.get_output_name(suffix='', prefix_log_filename=self.projector_key)),
                round_id=self.current_round
            )
        else:
            return None


class Client(BasicClient):
    def __init__(self,
    option, name='', train_data=None, valid_data=None, modalities=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.modalities = modalities
        self.projector_key = "+".join(self.modalities)

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
            "projector_key": self.projector_key
        }

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
        return batch_data