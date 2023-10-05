from typing import Any
import torch
from torch import nn
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
        in_size = 50
        hidden_size = 32    # FedMSplit
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

    
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(32, 2, True)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln(x)
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.feature_extractors = nn.ModuleList()
        for i in range(self.n_leads):
            self.feature_extractors.append(SubNet())
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, leads):
        # import pdb; pdb.set_trace()
        
        batch_size = y.shape[0]
        # print(batch_size)
        features = torch.zeros(size=(batch_size, 32), dtype=torch.float32, device=y.device)
        for lead in leads:
            features += self.feature_extractors[lead](x[:, lead, :])
        outputs = self.classifier(features)
        # import pdb; pdb.set_trace()
        loss = self.criterion(outputs, y.type(torch.int64))
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()