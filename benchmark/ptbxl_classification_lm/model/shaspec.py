from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
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

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(128*12, 128, True)
        self.ln2 = nn.Linear(128, 10, True)
    
    def forward(self, x):       #()
        return self.ln2(F.relu(self.ln1(x)))
    
class ModalityClassifier(FModule):
    def __init__(self, n_classes=12):
        super(ModalityClassifier, self).__init__()
        self.ln1 = nn.Linear(128, 128, True)
        self.ln2 = nn.Linear(128, n_classes, True)
    
    def forward(self, x):
        return self.ln2(F.relu(self.ln1(x)))
    
class ModalityProjector(FModule):
    def __init__(self):
        self.ln1 = nn.Linear(128*2, 128, False)
        self.ln2 = nn.Linear(128, 128, True)
        
    def forward(self, x):
        x = F.relu(self.ln1(x))
        return self.ln2(x)
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 12
        self.hidden_dim = 128
        self.specific_extractor = nn.ModuleList()
        self.modality_prototypes = nn.ModuleList()
        self.modality_projectors = nn.ModuleList()
        self.global_align_classifier = ModalityClassifier(n_clases=self.n_leads)
        for i in range(self.n_leads):
            self.specific_extractor.append(Inception1DBase(input_channels=1))
            self.modality_prototypes.append(nn.Parameter(torch.zeros(1, 128)))
            nn.init.uniform_(self.modality_prototypes[i], -1.0, 1.0)
            
            self.modality_projectors.append(ModalityProjector())
            
        self.local_shared_extractor = Inception1DBase(input_channels=1)
        self.global_shared_extractor = Inception1DBase(input_channels=1)
        self.global_align_classifier = ModalityClassifier(self.n_leads)
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def set_local_leads(self, leads):
        n_local_leads = len(leads)
        self.local_leads=leads
        self.local_align_classifier = ModalityClassifier(n_classes=n_local_leads)
        self.local_modality_classifier = ModalityClassifier(n_classes=n_local_leads)
        
    def forward(self, x, y, leads): # x: B x 12 x C
        assert(hasattr(self, 'local_align_classifier')), "Local Align Classifier must be initialized first"
        assert(hasattr(self, 'local_modality_classifier')), "Local Modality Classifier must be initialized first"
        
        batch_size = y.shape[0]
        total_lead_ind = [*range(12)]
        specific_features = []
        local_shared_features = []
        global_shared_features = []
                
        # Stage 1: Encoding
        for lead in total_lead_ind:    
            if lead in leads:   # include local missing lead
                
                input_data = x[:, lead, :].view(batch_size, 1, -1) + self.modality_prototypes[lead].repeat(batch_size, 1) # x + prototype
                specific_feature = self.specific_extractor[lead](input_data)  # B x 1 x C
                local_shared_feature = self.local_shared_extractor[lead](input_data) # B x 1 x C
                global_shared_feature = self.global_shared_extractor[lead](input_data) # B x 1 x C
                specific_features.append(specific_feature)
                local_shared_features.append(local_shared_feature)
                global_shared_features.append(global_shared_feature)
                
            else:
                input_data = self.modality_prototypes[lead].repeat(batch_size,1)       
                specific_feature = self.specific_extractor[lead](input_data) # B x 1 x C
                local_shared_feature = self.local_shared_extractor[lead](input_data)
                specific_features.append(specific_feature)
                local_shared_features.append(local_shared_feature)
                global_shared_features.append(global_shared_feature)
                
        # Step 2: Downstream task
        fuse_features = []
        for lead in total_lead_ind:
            if lead in leads:
                shared_features_list = local_shared_features
            else: 
                shared_features_list = global_shared_features
            shared_specific_feat = torch.cat([shared_features_list[lead], specific_features[lead]], dim=2)
            fuse_feature = self.modality_projectors[lead](shared_specific_feat) + local_shared_features[lead]
            fuse_features.append(fuse_feature)
        
                
        outputs = self.classifier(torch.cat(fuse_features, dim=2))
        downstream_loss = self.criterion(outputs, y.type(torch.int64))
        
        # Step 3: Local modality classification loss
        # input: Bx12xC - output: Bx12x12 
        local_cls_input = torch.cat(local_shared_features, dim=1)
        local_cls_label = torch.zeros((batch_size, 12, 12), device=y.device)
        for lead in total_lead_ind:
            local_cls_label[:, lead, lead] = 1.0
        local_cls_output = self.local_modality_classifier(local_cls_input)
        local_cls_loss = self.criterion(local_cls_output, local_cls_label)
        
        # Step 4: Local align classification loss
        local_aln_input = torch.cat(local_shared_features, dim=1)
        local_aln_label = torch.ones((batch_size, 12, 12), device=y.device) * 1/12
        local_aln_output = self.local_align_classifier(local_aln_input)
        local_aln_loss = self.criterion(local_aln_output, local_aln_label)   
        
        # Step 5: Global align classification loss
        global_aln_input = torch.cat(global_shared_features, dim=1)
        global_aln_label = torch.ones((batch_size, 12, 12), device=y.device) * 1/12
        global_aln_output = self.global_align_classifier(global_aln_input)
        global_aln_loss = self.criterion(global_aln_output, global_aln_label)
        
        loss = downstream_loss + 0.1*local_cls_loss + 0.1*local_aln_loss + 0.1*global_aln_loss
        loss_list = {
            'downstream': downstream_loss.cpu(),
            'local_cls': local_cls_loss.cpu(),
            'local_aln': local_aln_loss.cpu(),
            'global_aln': global_aln_loss.cpu()
        }
        return loss_list, loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()