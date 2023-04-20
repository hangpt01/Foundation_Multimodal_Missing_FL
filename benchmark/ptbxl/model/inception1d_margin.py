import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from benchmark.ptbxl.model.inception1d import (
    Inception1D2Leads,
    Inception1D3Leads,
    Inception1D4Leads,
    Inception1D6Leads,
    Inception1DAllLeads,
    Classifier
)

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
        self.criterion = torch.nn.BCEWithLogitsLoss()
    def forward(self, x, y, leads='all', contrastive_weight=0.0, temperature=0.0, margin=0.0):
        if leads == 'all':
            xallleads = self.branchallleads(x)
            outputs = self.classifier(xallleads)
            loss = self.criterion(outputs, y)
            # contrastive loss
            if contrastive_weight > 0.0:
                xallleads = F.normalize(xallleads, p=2, dim=1)
                x2leads = F.normalize(self.branch2leads(x[:, self.lead_idx['2']]), p=2, dim=1)
                positive_sim = torch.mul(xallleads, x2leads).sum(dim=-1)
                device = y.device
                margins = torch.full_like(positive_sim, fill_value=margin, device=device)
                contrastive_loss = 1.0 - torch.minimum(positive_sim, margins)
                loss += contrastive_weight * contrastive_loss.mean()
            return loss, outputs
        else:
            x2leads = self.branch2leads(x[:, self.lead_idx['2']])
            outputs = self.classifier(x2leads)
            loss = self.criterion(outputs, y)
            return loss, outputs