from typing import Any
import torch
from torch import nn
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


class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(128, 64, True)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(64, 6, True)
        
        # self.classifier = nn.Sequential(
            #     nn.Linear(d_hid*2, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, num_classes)
            # )
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        
        return self.ln2(self.relu(self.ln1(x)))
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.feature_extractors = nn.ModuleList()
        for i in range(self.n_leads):
            self.feature_extractors.append(HAR_Feature_Extractor())
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, leads, contrastive_weight=5):
        # import pdb; pdb.set_trace()
        
        batch_size = y.shape[0]
        hidden_dim = 128
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