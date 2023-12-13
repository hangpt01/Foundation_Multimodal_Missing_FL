from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class SubNet(FModule):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size=768):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
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
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1   


class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(64, 6, True)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln(x)
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 3        # t-a-v: text-audio-visual
        self.feature_extractors = nn.ModuleList()
        self.feature_extractors.append(TextSubNet())
        for i in range(2):
            self.feature_extractors.append(SubNet())
        
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, leads):
        # import pdb; pdb.set_trace()
        
        batch_size = y.shape[0]
        hidden_dim = 64
        # print(batch_size)
        features = torch.zeros(size=(batch_size, hidden_dim), dtype=torch.float32, device=y.device)
        for lead in leads:
            # input = x[:, lead, :]
            # features += self.feature_extractors[lead](input[input.nonzero()])
            features += self.feature_extractors[lead](x[:, lead, :])
        outputs = self.classifier(features)
        # import pdb; pdb.set_trace()
        loss = self.criterion(outputs, y.type(torch.int64))
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()