import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, input_channels):
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
        # x = F.normalize(x, p=2, dim=1)
        return x
        
class Inception1D2Leads(Inception1DBase):
    def __init__(self):
        super(Inception1D2Leads, self).__init__(input_channels=2)
        
class Inception1D3Leads(Inception1DBase):
    def __init__(self):
        super(Inception1D3Leads, self).__init__(input_channels=3)
        
class Inception1D4Leads(Inception1DBase):
    def __init__(self):
        super(Inception1D4Leads, self).__init__(input_channels=4)
        
class Inception1D6Leads(Inception1DBase):
    def __init__(self):
        super(Inception1D6Leads, self).__init__(input_channels=6)
        
class Inception1DAllLeads(Inception1DBase):
    def __init__(self):
        super(Inception1DAllLeads, self).__init__(input_channels=12)
        
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(128, 71, True)
    def forward(self, x):
        return self.ln(x)
        
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.lead_idx = {
            '2': [0, 1],
            '3': [0, 1, 7],
            '4': [0, 1, 2, 7],
            '6': [0, 1, 2, 3, 4, 5]
        }
        self.branchallleads = Inception1DAllLeads()
        self.branch2leads = Inception1D2Leads()
        self.classifier = Classifier()
        # self.branchallleads_classifier = Classifier()
        # self.branch2leads_classifier = Classifier()
        self.criterion = torch.nn.BCEWithLogitsLoss()
    def forward(self, x, y, leads='all', contrastive_weight=1.0, temperature=1.0):
        if leads == 'all':
            x2leads = self.branch2leads(x[:, self.lead_idx['2']])
            xallleads = self.branchallleads(x)
            outputs = self.classifier(xallleads)
            loss = self.criterion(outputs, y)
            # contrastive loss
            if contrastive_weight > 0.0:
                batch_size = y.shape[0]
                device = y.device
                concat_reprs = torch.concat((xallleads, x2leads), dim=0)
                exp_sim_matrix = torch.exp(torch.mm(concat_reprs, concat_reprs.t().contiguous()) / temperature)
                mask = (torch.ones_like(exp_sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
                exp_sim_matrix = exp_sim_matrix.masked_select(mask=mask).view(2 * batch_size, -1)
                positive_exp_sim = torch.exp(torch.sum(xallleads * x2leads, dim=-1) / temperature)
                positive_exp_sim = torch.concat((positive_exp_sim, positive_exp_sim), dim=0)
                contrastive_loss = - torch.log(positive_exp_sim / exp_sim_matrix.sum(dim=-1))
                loss += contrastive_weight * contrastive_loss.mean()
            return loss, outputs
        else:
            x2leads = self.branch2leads(x[:, self.lead_idx['2']])
            outputs = self.classifier(x2leads)
            loss = self.criterion(outputs, y)
            return loss, outputs
            



if __name__ == '__main__':
    model = Inception1DAllLeads()
    for name, parameter in model.named_parameters():
        # nn.init.ones_(parameter)
        parameter.data.fill_(0.01)
        # else:
        #     print(name)
    model.eval()
    x = torch.ones(128, 12, 250)
    y = model(x)
    print(y)
    print(y.sum())
    # x = torch.ones(128, 12, 250)
    # y = model(x)
    # print(y)
    # print(y.sum())
    # import pdb; pdb.set_trace()
    