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
        self.ln1 = nn.Linear(128*12, 128, True)
        # self.ln2 = nn.Linear(128, 5, True)
        self.ln2 = nn.Linear(128, 5, True)
    
    def forward(self, x):       #()
        return self.ln2(F.relu(self.ln1(x)))
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 12
        self.hidden_dim = 128
        self.feature_extractors = nn.ModuleList()
        self.relation_embedders = nn.ModuleList()
        for i in range(self.n_leads):
            self.feature_extractors.append(Inception1DBase(input_channels=1))
            self.relation_embedders.append(RelationEmbedder())
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y, leads, contrastive_weight):
        batch_size = y.shape[0]
        features = torch.zeros(size=(batch_size, self.hidden_dim*12), dtype=torch.float32, device=y.device)
        total_lead_ind = [*range(12)]
        leads_features = []
        feature_extractor_outputs = torch.zeros(size=(batch_size, self.hidden_dim), dtype=torch.float32, device=y.device)
        for lead in total_lead_ind:    
            if lead in leads:
                feature = self.feature_extractors[lead](x[:, lead, :].view(batch_size, 1, -1))
                leads_features.append(feature)
                feature_extractor_outputs += feature
                relation_info = self.relation_embedders[lead](y.device, has_modal=True).repeat(batch_size,1)
                feature = feature + relation_info
                features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
                self.relation_embedders[lead].relation_embedder.weight.data[0].zero_()
            else:
                feature = self.relation_embedders[lead](y.device, has_modal=False).repeat(batch_size,1)        # self.hidden_dim, 256
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
        # import pdb; pdb.set_trace()
        if count > 0:
            loss += contrastive_weight * contrative_loss / count

        # loss_leads = 0
        # return loss_leads, loss, outputs
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()