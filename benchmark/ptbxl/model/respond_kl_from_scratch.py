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
        return x
        
class Inception1D1Leads(Inception1DBase):
    def __init__(self):
        super(Inception1D1Leads, self).__init__(input_channels=1)
        
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
        

def multivariate_KL_divergence(teacher_batch_input, student_batch_input, device):
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    teacher_batch_input = teacher_batch_input.to(device).unsqueeze(1)
    student_batch_input = student_batch_input.to(device).unsqueeze(1)
    
    sub_s = student_batch_input - student_batch_input.transpose(0,1)
    sub_s_norm = torch.norm(sub_s, dim=2)
    sub_s_norm = sub_s_norm.flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t = teacher_batch_input - teacher_batch_input.transpose(0,1)
    sub_t_norm = torch.norm(sub_t, dim=2)
    sub_t_norm = sub_t_norm.flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.lead_idx = {
            # '2': [0, 1],
            '2': [8],
            '3': [0, 1, 7],
            '4': [0, 1, 2, 7],
            '6': [0, 1, 2, 3, 4, 5]
        }
        self.branchallleads = Inception1DAllLeads()
        # self.branch2leads = Inception1D2Leads()
        self.branch2leads = Inception1D1Leads()
        self.branchallleads_classifier = Classifier()
        self.branch2leads_classifier = Classifier()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.kl_div = torch.nn.KLDivLoss()
    def forward(self, x, y, leads='all', contrastive_weight=0.0, temperature=0.0, margin=0.0, kl_weight=0.0):
        if leads == 'all':
            xallleads = self.branchallleads(x)
            outputs_allleads = self.branchallleads_classifier(xallleads)
            loss_allleads = self.criterion(outputs_allleads, y)

            x2leads = self.branch2leads(x[:, self.lead_idx['2']])
            outputs_2leads = self.branch2leads_classifier(x2leads)
            loss_2leads = self.criterion(outputs_2leads, y)
            if kl_weight > 0.0:
                # kl_loss = multivariate_KL_divergence(xallleads, x2leads, xallleads.device)
                # loss_allleads += kl_weight * kl_loss
                tmp_p = outputs_allleads.clone().detach().view(-1)
                tmp_p = torch.unsqueeze(torch.sigmoid(tmp_p), dim=1)
                p = torch.cat((tmp_p, 1.0 - tmp_p), dim=1)

                tmp_q = outputs_2leads.view(-1)
                tmp_q = torch.unsqueeze(torch.sigmoid(tmp_q), dim=1)
                q = torch.cat((tmp_q, 1.0 - tmp_q), dim=1)

                kl_loss = self.kl_div(torch.log(q), p)
                loss_2leads += kl_weight * kl_loss

            return loss_allleads, outputs_allleads, loss_2leads, outputs_2leads
        else:
            x2leads = self.branch2leads(x[:, self.lead_idx['2']])
            outputs_2leads = self.branch2leads_classifier(x2leads)
            loss_2leads = self.criterion(outputs_2leads, y)

            return None, None, loss_2leads, outputs_2leads
            



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
    