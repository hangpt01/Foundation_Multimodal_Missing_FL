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

class RelationEmbedder(FModule):
    def __init__(self):
        super(RelationEmbedder, self).__init__()
        self.input_channels = 2     # Case 3
        self.relation_embedder = nn.Embedding(self.input_channels,32)
        nn.init.uniform_(self.relation_embedder.weight, -1.0, 1.0)

    def forward(self, device, has_modal=True):
        if has_modal:
            return self.relation_embedder(torch.tensor(1).to(device))
        else:
            return self.relation_embedder(torch.tensor(0).to(device))

    
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(32, 2, True)
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln(x)
    
    
# class Multi_Classifier(FModule):
#     def __init__(self):
#         super(Multi_Classifier, self).__init__()
#         self.ln1 = nn.Linear(32*2, 32, True)
#         self.ln2 = nn.Linear(32, 2, True)
    
#     def forward(self, x):       #()
#         return self.ln2(F.relu(self.ln1(x)))
    
class Multi_Classifier(FModule):
    def __init__(self):
        super(Multi_Classifier, self).__init__()
        self.ln = nn.Linear(32*2, 2, True)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln(x)

# class UniModel(FModule):
#     def __init__(self):
#         super(UniModel, self).__init__()
#         self.feature_extractor = Inception1DBase(input_channels=1)
#         self.relation_embedder = RelationEmbedder()
#         self.classifier = Classifier()
    
#     def forward(self, x, y, leads):       #()
#         return self.ln2(F.relu(self.ln1(x)))
    
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.feature_extractors = nn.ModuleList()
        self.relation_embedders = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.z_M = torch.FloatTensor([1/3]*3)
        self.delta_G = 0
        self.delta_O = 0
        self.hidden_dim = 32
        for i in range(self.n_leads):
            self.relation_embedders.append(RelationEmbedder())
            self.feature_extractors.append(SubNet())
            self.classifiers.append(Classifier())
        self.multi_classifier = Multi_Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y, leads):
        batch_size = y.shape[0]
        features = torch.zeros(size=(batch_size, self.hidden_dim*self.n_leads), dtype=torch.float32, device=y.device)
        total_lead_ind = [*range(self.n_leads)]
        # leads_features = []
        loss_leads = [0]*3
        # feature_extractor_outputs = torch.zeros(size=(batch_size, self.hidden_dim), dtype=torch.float32, device=y.device)
        # import pdb; pdb.set_trace()
        for lead in total_lead_ind:    
            if lead in leads:
                feature = self.feature_extractors[lead](x[:, lead, :])
                relation_info = self.relation_embedders[lead](y.device, has_modal=True).repeat(batch_size,1)
                feature = feature + relation_info
                output = self.classifiers[lead](feature)
                loss_leads[lead] = self.criterion(output,y.type(torch.int64))
                
                features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[0].zero_()
                
            else:
                feature = self.relation_embedders[lead](y.device, has_modal=False).repeat(batch_size,1)        # self.hidden_dim, 256
                output = self.classifiers[lead](feature)
                loss_leads[lead] = self.criterion(output,y.type(torch.int64))
                
                features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[1].zero_()
        
        # import pdb; pdb.set_trace()
        
        outputs_multi = self.multi_classifier(features)
        loss_multi = self.criterion(outputs_multi, y.type(torch.int64))
        loss_leads[-1] = loss_multi
        # import pdb; pdb.set_trace()
        loss =  sum([a*b for a,b in zip(self.z_M, loss_leads)])
    
        # outputs = self.classifier(features)
        # loss = self.criterion(outputs, y.type(torch.int64))

        return loss_leads, loss, outputs_multi

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()