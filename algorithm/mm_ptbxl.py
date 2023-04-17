from .fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.contrastive_weight = option['contrastive_weight']
        self.temperature = option['temperature']
        self.leads_cnt = {
            "all": {
                "n_clients": sum([1 for client in self.clients if client.modalities == "all"]),
                "n_data": sum([client.datavol for client in self.clients if client.modalities == "all"])
            },
            "2": {
                "n_clients": sum([1 for client in self.clients if client.modalities == "2"]),
                "n_data": sum([client.datavol for client in self.clients if client.modalities == "2"])
            }
        }

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
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, modalities_list)
        return

    def aggregate(self, models: list, modalities_list: list):
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
        new_model.classifier = fmodule._model_sum([model_k.classifier * pk for model_k, pk in zip(models, p)])
        new_model.branch2leads = fmodule._model_sum([model_k.branch2leads * pk for model_k, pk in zip(models, p)])
        if self.leads_cnt["all"]["n_clients"] == 0:
            return new_model
        p = list()
        chosen_models = list()
        for cid, model, modalities in zip(self.received_clients, models, modalities_list):
            if modalities == "all":
                p.append(self.local_data_vols[cid])
                chosen_models.append(model)
        if len(chosen_models) == 0:
            return new_model
        sump = sum(p)
        p = [1.0 * pk / sump for pk in p]
        new_model.branchallleads = fmodule._model_sum([model_k.branchallleads * pk for model_k, pk in zip(chosen_models, p)])
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
                model=model,
                dataset=self.test_data,
                contrastive_weight=self.contrastive_weight,
                temperature=self.temperature,
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
        for c in self.clients:
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.modalities = modalities
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
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(
                model=model,
                data=batch_data,
                leads=self.modalities,
                contrastive_weight=self.contrastive_weight,
                temperature=self.temperature
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
            leads=self.modalities,
            contrastive_weight=self.contrastive_weight,
            temperature=self.temperature,
            batch_size=self.test_batch_size
        )