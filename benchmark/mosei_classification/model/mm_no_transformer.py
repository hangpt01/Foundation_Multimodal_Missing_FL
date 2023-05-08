import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from itertools import chain, combinations
from benchmark.mosei_classification.model.transformer_networks import TransformerEncoder
import numpy as np

# 1-modality Extractor
class TextExtractor(FModule):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.input_dim = 300
        self.hidden_dim = 30
        self.latent_dim = 60
        self.timestep = 50
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=False)
        # self.projector = nn.Linear(self.hidden_dim*self.timestep, self.latent_dim)

    def forward(self, x):       # (snapshot:50,300)
        batch = len(x)
        # import pdb; pdb.set_trace()
        input = x.reshape(batch, self.timestep, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return output.flatten(start_dim=1)
    
    
class AudioExtractor(FModule):
    def __init__(self):
        super(AudioExtractor, self).__init__()
        self.input_dim = 74
        self.hidden_dim = 30
        self.latent_dim = 60
        self.timestep = 50
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=False)
        # self.projector = nn.Linear(self.hidden_dim*self.timestep, self.latent_dim)

    def forward(self, x):       # (snapshot:50,300)
        batch = len(x)
        import pdb; pdb.set_trace()
        input = x.reshape(batch, self.timestep, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return output.flatten(start_dim=1)

class VisionExtractor(FModule):
    def __init__(self):
        super(VisionExtractor, self).__init__()
        self.input_dim = 35
        self.hidden_dim = 30
        self.latent_dim = 60
        self.timestep = 50
        
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=False)
        # self.projector = nn.Linear(self.hidden_dim*self.timestep, self.latent_dim)

    def forward(self, x):       # (snapshot:50,300)
        batch = len(x)
        # import pdb; pdb.set_trace()
        input = x.reshape(batch, self.timestep, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return output.flatten(start_dim=1)


# 1-modality Projector
class TextProjector(FModule):
    def __init__(self):
        super(TextProjector, self).__init__()
        self.ln = nn.Linear(30*50, 60)
    def forward(self, x):
        return self.ln(x)
    
class AudioProjector(FModule):
    def __init__(self):
        super(AudioProjector, self).__init__()
        self.ln = nn.Linear(30*50, 60)
    def forward(self, x):
        return self.ln(x)
    
class VisionProjector(FModule):
    def __init__(self):
        super(VisionProjector, self).__init__()
        self.ln = nn.Linear(30*50, 60)
    def forward(self, x):
        return self.ln(x)
    
    
# 2-modality Projector
class TextAudioProjector(FModule):
    def __init__(self):
        super(TextAudioProjector, self).__init__()
        self.ln = nn.Linear(30*50*2, 60, True)
    def forward(self, x):
        return self.ln(x)
    
class TextVisionProjector(FModule):
    def __init__(self):
        super(TextVisionProjector, self).__init__()
        self.ln = nn.Linear(30*50*2, 60, True)

    def forward(self, x):
        return self.ln(x)
    
class AudioVisionProjector(FModule):
    def __init__(self):
        super(AudioVisionProjector, self).__init__()
        self.ln = nn.Linear(30*50*2, 60, True)

    def forward(self, x):
        return self.ln(x)


# 3-modality Projector
class TextAudioVisionProjector(FModule):
    def __init__(self):
        super(TextAudioVisionProjector, self).__init__()
        self.ln = nn.Linear(30*50*3, 60, True)

    def forward(self, x):
        return self.ln(x)

# Multimodal Transformer
def get_affect_network(self_type='l', layers=1):
    if self_type in ['l', 'al', 'vl']:
        embed_dim, attn_dropout = 30, 0.1
    elif self_type in ['a', 'la', 'va']:
        embed_dim, attn_dropout = 30, 0.0
    elif self_type in ['v', 'lv', 'av']:
        embed_dim, attn_dropout = 30, 0.0
    elif self_type == 'l_mem':
        embed_dim, attn_dropout =  30, 0.1
    elif self_type == 'a_mem':
        embed_dim, attn_dropout =  30, 0.1
    elif self_type == 'v_mem':
        embed_dim, attn_dropout =  30, 0.1
    else:
        raise ValueError("Unknown network type")

    return TransformerEncoder(embed_dim=embed_dim,
                              num_heads=5,
                              layers=min(5, layers),
                              attn_dropout=attn_dropout,
                              relu_dropout=0.1,
                              res_dropout=0.1,
                              embed_dropout=0.25,
                              attn_mask=True)

class MultimodalTransformer(FModule):
    def __init__(self):
        super(MultimodalTransformer, self).__init__()
        self.common_dim = 60
        
        # language
        self.proj_l = nn.Conv1d(300, 30, kernel_size=1, padding=0, bias=False)
        self.trans_l_with_v = get_affect_network(self_type='lv', layers=5)
        self.trans_l_mem = get_affect_network(self_type='l_mem', layers=5)

        # # Audio
        # self.proj_a = nn.Conv1d(74, 30, kernel_size=1, padding=0, bias=False)
        # self.trans_a_with_l = get_affect_network(self_type='al', layers=5)
        # self.trans_a_with_v = get_affect_network(self_type='av', layers=5)
        # self.trans_a_mem = get_affect_network(self_type='a_mem', layers=5)

        # Vision
        self.proj_v = nn.Conv1d(35, 30, kernel_size=1, padding=0, bias=False)
        self.trans_v_with_l = get_affect_network(self_type='vl', layers=5)
        self.trans_v_mem = get_affect_network(self_type='v_mem', layers=5)

        # Projector
        self.proj1 = nn.Linear(60, 60)
        self.proj2 = nn.Linear(60, 60)
        # self.projector = nn.Linear(60, self.common_dim)
        

    def forward(self, x_l, x_v, name, iter):
        # x_l, x_v = x[:,:60], x[:,60:]
        # x_l, x_v = x[:,:1500], x[:,1500:]

        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # if name == 'Client00' and iter == 3:
        #     import pdb; pdb.set_trace()
        x_l = F.dropout(x_l.transpose(1, 2), p=0.25, training=self.training)
        # x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l)
        proj_x_v = self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V) --> L
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, name, iter)  # Dimension (L, N, d_l)
        h_ls = self.trans_l_mem(h_l_with_vs)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]  # Take the last output for prediction

        # (L) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_vs = self.trans_v_mem(h_v_with_ls)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # Concatenate
        last_hs = torch.cat([last_h_l, last_h_v], dim=1)
        # import pdb; pdb.set_trace()
        last_hs = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=0.0, training=self.training))
        # last_hs_proj += last_hs
        # import pdb; pdb.set_trace()
        # Project
        return last_hs      # (64,60)

class MultimodalProjector(FModule):     # (text+vision)
    def __init__(self):
        super(MultimodalProjector, self).__init__()
        self.ln = nn.Linear(60, 60)
    def forward(self, x):
        return self.ln(x)
    

class Regressor(FModule):
    def __init__(self):
        super(Regressor, self).__init__()
        self.ln = nn.Linear(60, 1, True)
    def forward(self, x):
        return torch.squeeze(self.ln(x))

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.modalities = ["text", "vision"]
        self.combin = "+".join(self.modalities)
        # self.modalities = ["text", "audio", "vision"]

        # feature extractors
        self.feature_extractors = nn.ModuleDict({
            "text": TextExtractor(),
            # "audio": AudioExtractor(),
            "vision": VisionExtractor()
        })
        
        # projectors
        self.projectors = nn.ModuleDict({
            "text": TextProjector(),
            # "audio": AudioProjector(),
            "vision": VisionProjector(),
            # "text+audio": TextAudioProjector(),
            "text+vision": TextVisionProjector(),
            # "text+vision": MultimodalProjector(),
            # "audio+vision": AudioVisionProjector(),
            # "text+audio+vision": TextAudioVisionProjector()
        })

        # self.multimodal_transformer = MultimodalTransformer()
        # regressor
        self.regressor = Regressor()

        # criterion
        self.L1Loss = nn.L1Loss()

    def forward(self, samples, labels, contrastive_weight=0.0, temperature=1.0, name=None, iter=None):
        current_modalities = samples.keys()
        if len(current_modalities) == 1 and 'vision' in current_modalities:
            features = self.feature_extractors['vision'](samples['vision'])
            representations = F.normalize(F.relu(self.projectors['vision'](features)), p=2, dim=1)
            outputs = self.regressor(representations)
            loss = self.L1Loss(outputs, labels)
        else:
            representations_dict = dict()
            features_dict = dict()
            for modal in self.modalities:
                features = self.feature_extractors[modal](samples[modal])
                features_dict[modal] = features
                representations_dict[modal] = F.normalize(F.relu(self.projectors[modal](features)), p=2, dim=1)
            joint_representations = F.normalize(F.relu(self.projectors[self.combin](torch.concat(tuple(features_dict.values()), dim=1))), p=2, dim=1)
            outputs = self.classifier(joint_representations)
            loss = self.CELoss(outputs, labels)
            batch_size = labels.shape[0]
            device = labels.device
            if batch_size > 1 and contrastive_weight > 0.0:
                contrastive_loss = 0.0
                concat_reprs = torch.concat((joint_representations, representations_dict['vision']), dim=0)
                exp_sim_matrix = torch.exp(torch.mm(concat_reprs, concat_reprs.t().contiguous()) / temperature)
                mask = (torch.ones_like(exp_sim_matrix) - torch.eye(2 * batch_size, device=device)).bool()
                exp_sim_matrix = exp_sim_matrix.masked_select(mask=mask).view(2 * batch_size, -1)
                positive_exp_sim = torch.exp(torch.sum(joint_representations * representations_dict['vision'], dim=-1) / temperature)
                positive_exp_sim = torch.concat((positive_exp_sim, positive_exp_sim), dim=0)
                contrastive_loss += - torch.log(positive_exp_sim / exp_sim_matrix.sum(dim=-1))
                loss += contrastive_weight * contrastive_loss.mean()
        return loss, outputs

if __name__ == '__main__':
    model = Model()
    # with open("./model.txt","w") as f:
    #     f.write(str(model))
    # print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()
    samples = {
        'text': torch.rand(size=(64, 50, 300)),
        'vision': torch.rand(size=(64, 50, 35))
    }
    labels = torch.rand(size=(64,))