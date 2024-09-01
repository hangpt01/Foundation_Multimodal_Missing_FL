from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from transformers import BertTokenizer  # Assuming text_embeddings from a transformer model
from utils.fmodule import FModule

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

### 1. **Main Network (Classifier)**

class MainNetwork(FModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MainNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### 2. **Reconstruction Network with Variance Output**

class ReconstructionNetwork(FModule):
    def __init__(self, input_dim, hidden_dim, num_priors):
        super(ReconstructionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, num_priors)  # Only outputs log variance
        
        self.apply(weights_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logvar = self.fc_logvar(h)
        var = torch.exp(logvar)  # Convert log variance to variance
        return var

    def reconstruct(self, x, modality_priors):
        """
        Reconstruct the missing modality using variance and modality priors.
        """
        var = self.forward(x)
        
        # Sample weights from Gaussian with mean I (identity matrix) and variance output by the network
        mean = torch.ones_like(var)  # Mean is identity (I)
        weights = mean + torch.sqrt(var) * torch.randn_like(var)  # Gaussian sampling with mean=1
        
        # Compute the weighted sum of the modality priors
        reconstructed_modality = torch.matmul(weights, modality_priors)
        return reconstructed_modality, weights, var

### 3. **Regularization Network**

class RegularizationNetwork(FModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegularizationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        return mean, std


### 5. **Modality Priors Initialization**

def initialize_modality_priors(data, num_priors):
    """
    Initialize modality priors using K-means clustering on complete modality data.
    """
    kmeans = KMeans(n_clusters=num_priors)
    kmeans.fit(data)
    modality_priors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return modality_priors


class Pooler(FModule):
    def __init__(self):
        super().__init__()
        hidden_size = 768
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        cls_num = 8
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, cls_num),
        )

    def forward(self, x):
        return self.classifier(x)

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.hparams_config = {'hidden_size': 768, 'max_image_len': 40}
        input_dim = 768
        hidden_dim = 768
        num_priors = 10
        output_dim = 8
        
        self.device = None
        self.transformer = None
        self.text_embeddings = None

        self.token_type_embeddings = nn.Embedding(2, self.hparams_config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        
        for param in self.token_type_embeddings.parameters():
            param.requires_grad = False

        self.pooler = Pooler()
        self.pooler.apply(init_weights)

        self.classifier = Classifier()
        self.classifier.apply(init_weights)

        self.main_network = MainNetwork(input_dim, hidden_dim, output_dim)
        self.reconstruction_network = ReconstructionNetwork(input_dim, hidden_dim, num_priors)
        self.regularization_network = RegularizationNetwork(input_dim, hidden_dim, input_dim)
        
        self.modality_priors = None  # Modality priors learned from complete samples
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps  # Number of gradient steps in the inner loop

        self.optimizer = optim.Adam(
            list(self.main_network.parameters()) +
            list(self.reconstruction_network.parameters()) +
            list(self.regularization_network.parameters()), lr=self.outer_lr
        )
        

    def infer(self, batch, transformer, text_embeddings, mask_text=False, mask_image=False, image_token_type_idx=1):
        self.transformer = transformer
        self.text_embeddings = text_embeddings

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]
        self.device = img.device

        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            img,
            max_image_len=self.hparams_config["max_image_len"],
            mask_it=mask_image,
        )

        if mask_text:
            # Reconstruct text features if missing
            text_feats_reconstructed = self.text_reconstruction(image_embeds.mean(dim=1))
            text_feats_reconstructed = self.text_regularization(text_feats_reconstructed)
            text_embeds = text_feats_reconstructed.unsqueeze(1).repeat(1, text_masks.size(1), 1)
        
        if mask_image:
            # Reconstruct image features if missing
            image_feats_reconstructed = self.image_reconstruction(text_embeds.mean(dim=1))
            image_feats_reconstructed = self.image_regularization(image_feats_reconstructed)
            image_embeds = image_feats_reconstructed.unsqueeze(1).repeat(1, image_masks.size(1), 1)

        # Combine real and reconstructed features
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))

        # Concatenate embeddings and masks
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        # Pass through transformer
        x = co_embeds.detach()

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)

        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, transformer, text_embeddings, batch, missing_text=False, missing_image=False):
        infer = self.infer(batch, transformer, text_embeddings, mask_text=missing_text, mask_image=missing_image)
        logits = self.classifier(infer["cls_feats"])

        imgcls_labels = batch["label"]
        imgcls_labels = torch.tensor(imgcls_labels).to(self.device).long()
        loss = F.cross_entropy(logits, imgcls_labels)
        return loss, loss, logits



if __name__ == '__main__':
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy input data for demonstration
    batch_size = 4
    image = torch.rand(batch_size, 3, 224, 224)  # Batch of images
    text = ["This is a sample text."] * batch_size  # Batch of text

    # Tokenizer for text inputs
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Labels for binary classification
    labels = torch.randint(0, 2, (batch_size,))

    # Dummy data loader
    data_loader = [(image, text_inputs['input_ids'], labels, 'text')]  # Example batch with missing text modality

    # Load sound mean (pre-calculated or dynamically calculated)
    sound_mean = torch.rand(768, 768)  # Example sound mean for KMeans

    # Meta-train the model with KMeans
    meta_train_with_kmeans(model, data_loader, optimizer, sound_mean, meta_lr=1e-3, inner_steps=1, n_clusters=10)
