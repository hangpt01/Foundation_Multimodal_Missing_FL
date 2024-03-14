from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViltModel
import numpy as np
from utils.fmodule import FModule

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 101))
    
    def forward(self, x):
        return self.classifier(x)

    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 2
        self.hidden_size = 768
        
        self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, leads, contrastive_weight):
        import pdb; pdb.set_trace()
        image = x['image']
        text = x['text']
        y = x['label']
        
        batch_size = y.shape[0]
        
        features = self.backbone(*x)
        outputs = self.classifier(features.last_hidden_state[:, 0, :])
        loss = self.criterion(outputs, y.type(torch.int64))
        
        # features = torch.zeros(size=(batch_size, self.hidden_dim*12), dtype=torch.float32, device=y.device)
        # total_lead_ind = [*range(12)]
        # leads_features = []
        # feature_extractor_outputs = torch.zeros(size=(batch_size, self.hidden_dim), dtype=torch.float32, device=y.device)
        # for lead in total_lead_ind:    
        #     if lead in leads:
        #         feature = self.feature_extractors[lead](x[:, lead, :].view(batch_size, 1, -1))
        #         leads_features.append(feature)
        #         feature_extractor_outputs += feature
        #         relation_info = self.relation_embedders[lead](y.device, has_modal=True).repeat(batch_size,1)
        #         feature = feature + relation_info
        #         features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
        #         self.relation_embedders[lead].relation_embedder.weight.data[0].zero_()
        #     else:
        #         feature = self.relation_embedders[lead](y.device, has_modal=False).repeat(batch_size,1)        # self.hidden_dim, 256
        #         features[:,lead*self.hidden_dim:(lead+1)*self.hidden_dim] = feature
        #         self.relation_embedders[lead].relation_embedder.weight.data[1].zero_()
        # outputs = self.classifier(features)
        # loss = self.criterion(outputs, y.type(torch.int64))


        # labels = y.cpu().numpy().astype(np.int64)
        # unique_labels = np.unique(labels)
        # norm_features = F.normalize(feature_extractor_outputs, p=2, dim=1)
        # contrative_loss = 0.0
        # count = 0
        # for lead_features in leads_features:
        #     norm_lead_features = F.normalize(lead_features, p=2, dim=1)
        #     simi_mat = norm_features.matmul(norm_lead_features.T)
        #     exp_simi_mat = torch.exp(simi_mat / 1.0)
        #     for label in unique_labels:
        #         positive_idx = np.where(labels == label)[0]
        #         negative_idx = np.where(labels != label)[0]
        #         positive = exp_simi_mat.diagonal()[positive_idx]
        #         negative = exp_simi_mat[positive_idx, :][:, negative_idx].sum(dim=1)
        #         contrative_loss -= torch.log(positive / (positive + negative)).sum()
        #         negative = exp_simi_mat[negative_idx, :][:, positive_idx].sum(dim=0)
        #         contrative_loss -= torch.log(positive / (positive + negative)).sum()
        #         count += positive_idx.shape[0] * 2
        # # import pdb; pdb.set_trace()
        # if count > 0:
        #     loss += contrastive_weight * contrative_loss / count

        # loss_leads = 0
        # return loss_leads, loss, outputs
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    import pdb; pdb.set_trace()