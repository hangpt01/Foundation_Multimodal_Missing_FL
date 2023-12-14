from typing import Any
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils.fmodule import FModule

class SubNet(FModule):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        in_size=768
        hidden_size = 64    # FedMSplit
        dropout = 0
        super(SubNet, self).__init__()
        self.bnorm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # import pdb; pdb.set_trace()        
        normed = self.bnorm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

class TextSubNet(FModule):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        in_size = 768
        hidden_size = 64
        dropout = 0
        out_size = hidden_size
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=1, dropout=dropout, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        batch_size = x.shape[0]
        _, final_states = self.rnn(x.view(batch_size, 1, -1))
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1   


class RelationEmbedder(FModule):
    def __init__(self):
        super(RelationEmbedder, self).__init__()
        self.input_channels = 2     # Case 3
        self.relation_embedder = nn.Embedding(self.input_channels,64)
        nn.init.uniform_(self.relation_embedder.weight, -1.0, 1.0)

    def forward(self, device, has_modal=True):
        if has_modal:
            return self.relation_embedder(torch.tensor(1).to(device))
        else:
            return self.relation_embedder(torch.tensor(0).to(device))

    
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(64*3, 6, True)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln(x)
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 3        # t-a-v: text-audio-visual
        self.hidden_dim = 64
        self.feature_extractors = nn.ModuleList()
        self.relation_embedders = nn.ModuleList()

        self.feature_extractors.append(TextSubNet())
        self.relation_embedders.append(RelationEmbedder())

        for i in range(2):
            self.feature_extractors.append(SubNet())
            self.relation_embedders.append(RelationEmbedder())
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, leads):
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
            loss += 5.0 * contrative_loss / count

        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()