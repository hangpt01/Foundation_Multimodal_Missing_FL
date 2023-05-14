from utils import fmodule
import copy
import utils.fflow as flw
import utils.system_simulator as ss
from .fedbase import BasicServer, BasicClient
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
        self.kl_weight = option['kl_weight']
        # self.all_modal_combin_list = list()
        # for combin_tuple in chain.from_iterable(combinations(self.all_modalities, r) for r in range(1, len(self.all_modalities) + 1)):
        #     self.all_modal_combin_list.append(list(combin_tuple))
        # print(self.all_modal_combin_list)
        self.combin_cnt = {
            "text+vision": {
                "n_clients": sum([1 for client in self.clients if client.modal_combin == "text+vision"]),
                "n_data": sum([client.datavol for client in self.clients if client.modal_combin == "text+vision"])
            },
            "vision": {
                "n_clients": sum([1 for client in self.clients if client.modal_combin == "vision"]),
                "n_data": sum([client.datavol for client in self.clients if client.modal_combin == "vision"])
            }
        }
        print(self.combin_cnt)

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
        # print("TRAININGGGGGGG")
        conmmunitcation_result = self.communicate(self.selected_clients)
        # print("DONEEEEE")
        models = conmmunitcation_result['model']
        # modalities_list = conmmunitcation_result['modalities']
        modal_combins = conmmunitcation_result['modal_combin']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # self.model = self.aggregate(models, modalities_list, modal_combins)
        # self.model = models[0]
        self.model = self.aggregate(models, modal_combins)
        # if self.current_round % 5 == 0:
        #     torch.save(self.model.state_dict(), os.path.join(self.chkpt_dir, 'Round{}.pt'.format(self.current_round)))
        return

    def aggregate(self, models: list, modal_combins: list):
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

        p = [self.local_data_vols[cid] for cid in self.received_clients]
        sump = sum(p)
        p = [1.0 * pk / sump for pk in p]
        new_model.vision_encoder = fmodule._model_sum([model_k.vision_encoder * pk for model_k, pk in zip(models, p)])
        new_model.vision_regressor = fmodule._model_sum([model_k.vision_regressor * pk for model_k, pk in zip(models, p)])
        if self.combin_cnt["text+vision"]["n_clients"] > 0:
            p = list()
            chosen_models = list()
            for cid, model, combin in zip(self.received_clients, models, modal_combins):
                if combin == "text+vision":
                    p.append(self.local_data_vols[cid])
                    chosen_models.append(model)
            if len(chosen_models) > 0:
                sump = sum(p)
                p = [1.0 * pk / sump for pk in p]
                new_model.joint_encoder = fmodule._model_sum([model_k.joint_encoder * pk for model_k, pk in zip(models, p)])
                new_model.joint_regressor = fmodule._model_sum([model_k.joint_regressor * pk for model_k, pk in zip(models, p)])
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
            import pdb; pdb.set_trace()
            return self.calculator.custom_test(
                model=model,
                dataset=self.test_data,
                contrastive_weight=self.contrastive_weight,
                temperature=self.temperature,
                margin=self.margin,
                kl_weight=self.kl_weight,
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
        # return dict()
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self,
    option, name='', train_data=None, valid_data=None, modalities=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.modalities = modalities
        self.modal_combin = "+".join(self.modalities)
        self.contrastive_weight = option['contrastive_weight']
        self.temperature = option['temperature']
        self.margin = option['margin']
        self.kl_weight = option['kl_weight']

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
            loss = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                modalities=self.modalities,
                contrastive_weight=self.contrastive_weight,
                temperature=self.temperature,
                margin=self.margin,
                kl_weight=self.kl_weight
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
            modalities=self.modalities,
            contrastive_weight=self.contrastive_weight,
            temperature=self.temperature,
            margin=self.margin,
            kl_weight=self.kl_weight,
            batch_size=self.test_batch_size
        )

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