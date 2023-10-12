from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.fmodule import FModule

class InceptionBlock1D(FModule):
    def __init__(self, input_channels):
        super(InceptionBlock1D, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=19, stride=1, padding=9, bias=False)
        self.convs_conv3 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.convbottle_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.convbottle_conv = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.bnrelu_bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bnrelu_relu = nn.ReLU()
    def forward(self, x):
        bottled = self.bottleneck(x)
        y = torch.cat([
            self.convs_conv1(bottled),
            self.convs_conv2(bottled),
            self.convs_conv3(bottled),
            self.convbottle_conv(self.convbottle_maxpool(x))
        ], dim=1)
        out = self.bnrelu_relu(self.bnrelu_bn(y))
        return out

class Shortcut1D(FModule):
    def __init__(self, input_channels):
        super(Shortcut1D, self).__init__()
        self.input_channels = input_channels
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(self.input_channels, 128, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))
        
class Inception1DBase(FModule):
    def __init__(self, input_channels=1):
        super(Inception1DBase, self).__init__()
        self.input_channels = input_channels
        # inception backbone
        self.inceptionbackbone_1 = InceptionBlock1D(input_channels=self.input_channels)
        self.inceptionbackbone_2 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_3 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_4 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_5 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_6 = InceptionBlock1D(input_channels=128)
        # shortcut
        self.shortcut_1 = Shortcut1D(input_channels=self.input_channels)
        self.shortcut_2 = Shortcut1D(input_channels=128)
        # pooling
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.mp = nn.AdaptiveMaxPool1d(output_size=1)
        # flatten
        self.flatten = nn.Flatten()
        self.bn_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_1 = nn.Dropout(p=0.25, inplace=False)
        self.ln_1 = nn.Linear(256, 128, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_2 = nn.Dropout(p=0.5, inplace=False)
        # self.ln_2 = nn.Linear(128, 71, bias=True)
    def forward(self, x):
        # inception backbone
        input_res = x
        x = self.inceptionbackbone_1(x)
        x = self.inceptionbackbone_2(x)
        x = self.inceptionbackbone_3(x)
        x = self.shortcut_1(input_res, x)
        input_res = x.clone()
        x = self.inceptionbackbone_4(x)
        x = self.inceptionbackbone_5(x)
        x = self.inceptionbackbone_6(x)
        x = self.shortcut_2(input_res, x)
        # input_res = x.clone()
        # head
        x = torch.cat([self.mp(x), self.ap(x)], dim=1)
        x = self.flatten(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)
        x = self.ln_1(x)
        x = self.relu(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)
        return x

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
        self.ln = nn.Linear(128, 10, True)
    
    def forward(self, x):
        return self.ln(x)
    
    
class Multi_Classifier(FModule):
    def __init__(self):
        super(Multi_Classifier, self).__init__()
        self.ln1 = nn.Linear(128*12, 128, True)
        self.ln2 = nn.Linear(128, 10, True)
    
    def forward(self, x):       #()
        return self.ln2(F.relu(self.ln1(x)))

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
        self.n_leads = 12
        self.feature_extractors = nn.ModuleList()
        self.relation_embedders = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.z_M = torch.FloatTensor([1/13]*13)
        self.delta_G = 0
        self.delta_O = 0
        # self.p_k = 0
        for i in range(self.n_leads):
            self.feature_extractors.append(Inception1DBase(input_channels=1))
            self.relation_embedders.append(RelationEmbedder())
            self.classifiers.append(Classifier())
        self.multi_classifier = Multi_Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y, leads):
        batch_size = y.shape[0]
        features = torch.zeros(size=(batch_size, 128*12), dtype=torch.float32, device=y.device)
        total_lead_ind = [*range(12)]
        # leads_features = []
        loss_leads = [0]*13
        # feature_extractor_outputs = torch.zeros(size=(batch_size, 128), dtype=torch.float32, device=y.device)
        # import pdb; pdb.set_trace()
        for lead in total_lead_ind:    
            if lead in leads:
                feature = self.feature_extractors[lead](x[:, lead, :].view(batch_size, 1, -1))
                relation_info = self.relation_embedders[lead](y.device, has_modal=True).repeat(batch_size,1)
                feature = feature + relation_info
                output = self.classifiers[lead](feature)
                loss_leads[lead] = self.criterion(output,y.type(torch.int64))
                
                features[:,lead*128:(lead+1)*128] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[0].zero_()
                
            else:
                feature = self.relation_embedders[lead](y.device, has_modal=False).repeat(batch_size,1)        # 128, 256
                output = self.classifiers[lead](feature)
                loss_leads[lead] = self.criterion(output,y.type(torch.int64))
                
                features[:,lead*128:(lead+1)*128] = feature
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