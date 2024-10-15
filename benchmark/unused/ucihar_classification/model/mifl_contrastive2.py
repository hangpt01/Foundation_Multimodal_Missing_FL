from typing import Any
import torch
from torch import nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from utils.fmodule import FModule

class Conv1dEncoder(FModule):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class HAR_Feature_Extractor(FModule):
    def __init__(
        self, 
        num_classes: int=6,       # Number of classes 
        acc_input_dim: int=3,     # Acc data input dim
        gyro_input_dim: int=3,    # Gyro data input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(HAR_Feature_Extractor, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )
        # projection
        self.acc_proj = nn.Linear(d_hid, d_hid//2)

        self.init_weight()


    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc):
        # x_acc: (batch, dim=128*3)
        # requires: (batch, dim, 3)
        # import pdb; pdb.set_trace()
        x_acc = self.acc_conv(x_acc.reshape(x_acc.shape[0], -1, 3))
        x_acc, _ = self.acc_rnn(x_acc)

        x_acc = torch.mean(x_acc, axis=1)
        # x_gyro = torch.mean(x_gyro, axis=1)
        # x_mm = torch.cat((x_acc, x_gyro), dim=1)

        # # 5. Projection
        # if self.en_att and self.att_name != "fuse_base":
        #     x_acc = self.acc_proj(x_acc)
        #     x_gyro = self.gyro_proj(x_gyro)
        #     x_mm = torch.cat((x_acc, x_gyro), dim=1)
        
        # # 6. MM embedding and predict
        # preds = self.classifier(x_mm)
        # return preds, x_mm
        return x_acc


class RelationEmbedder(FModule):
    def __init__(self):
        super(RelationEmbedder, self).__init__()
        self.input_channels = 2     # Case 3
        self.relation_embedder = nn.Embedding(self.input_channels,128)
        nn.init.uniform_(self.relation_embedder.weight, -1.0, 1.0)

    def forward(self, device, has_modal=True):
        if has_modal:
            return self.relation_embedder(torch.tensor(1).to(device))
        else:
            return self.relation_embedder(torch.tensor(0).to(device))

    
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(128*2, 64, True)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(64, 6, True)
    
    def forward(self, x):
        return self.ln2(self.relu(self.ln1(x)))
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.hidden_dim = 128
        self.relation_embedders = nn.ModuleList()
        self.feature_extractors = nn.ModuleList()
        for i in range(self.n_leads):
            self.feature_extractors.append(HAR_Feature_Extractor())
            self.relation_embedders.append(RelationEmbedder())
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, leads, contrastive_weight):
        batch_size = y.shape[0]
        features = torch.zeros(size=(batch_size, self.hidden_dim*self.n_leads), dtype=torch.float32, device=y.device)
        total_lead_ind = [*range(self.n_leads)]
        leads_features = []
        feature_extractor_outputs = torch.zeros(size=(batch_size, self.hidden_dim), dtype=torch.float32, device=y.device)
        for lead in total_lead_ind:    
            if lead in leads:
                feature = self.feature_extractors[lead](x[:, lead, :].view(batch_size, -1))
                leads_features.append(feature)
                feature_extractor_outputs += feature
                relation_info = self.relation_embedders[lead](y.device, has_modal=True).repeat(batch_size,1)
                feature = feature + relation_info
                features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[0].zero_()
            else:
                feature = self.relation_embedders[lead](y.device, has_modal=False).repeat(batch_size,1)        # self.hidden_dim, 256
                feature_extractor_outputs += feature
                features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[1].zero_()
        outputs = self.classifier(features)
        loss = self.criterion(outputs, y.type(torch.int64))


        labels = y.cpu().numpy().astype(np.int64)
        unique_labels = np.unique(labels)
        norm_features = F.normalize(feature_extractor_outputs, p=2, dim=1)
        contrative_loss = 0.0
        count = 0
        for lead_features in leads_features:
            norm_lead_features = F.normalize(lead_features, p=2, dim=1)
            simi_mat = norm_features.matmul(norm_lead_features.T)
            exp_simi_mat = torch.exp(simi_mat / 1.0)
            for label in unique_labels:
                positive_idx = np.where(labels == label)[0]
                negative_idx = np.where(labels != label)[0]
                positive = exp_simi_mat.diagonal()[positive_idx]
                negative = exp_simi_mat[positive_idx, :][:, negative_idx].sum(dim=1)
                contrative_loss -= torch.log(positive / (positive + negative)).sum()
                negative = exp_simi_mat[negative_idx, :][:, positive_idx].sum(dim=0)
                contrative_loss -= torch.log(positive / (positive + negative)).sum()
                count += positive_idx.shape[0] * 2
        if count > 0:
            loss += contrastive_weight * contrative_loss / count

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()