import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule


class Img_FC (FModule):
    def __init__(self):
        super(Img_FC, self).__init__()
        img_input_dim = 1280
        d_hid = 64
        self.fc = nn.Linear(img_input_dim, d_hid)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class Text_FC (FModule):
    def __init__(self):
        super(Text_FC, self).__init__()
        text_input_dim = 512
        d_hid = 64
        self.fc = nn.Linear(text_input_dim, d_hid)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class ModelAttention (FModule):
    def __init__(self):
        super(ModelAttention, self).__init__()
        d_hid = 64
        self.att = nn.MultiheadAttention(embed_dim=d_hid, num_heads=4, batch_first=True)
    
    def forward(self,x):
        return self.att(x, x, x)
    
class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        cls_num = 101
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
        """
        Args:
            num_classes (int): Number of output classes for classification.
            img_input_dim (int): Dimension of precomputed image features (e.g., MobileNetV2 output size).
            text_input_dim (int): Dimension of precomputed text features (e.g., MobileBERT output size).
            d_hid (int): Hidden size of intermediate layers.
            en_att (bool): Whether to use attention-based fusion.
            att_name (str): Type of attention mechanism ('multihead' or other).
        """
        super(Model, self).__init__()
        
        # Image feature processing
        self.img_fc = Img_FC()
        self.img_fc.m = None
        self.img_fc.v = None
        
        # Text feature processing
        self.text_fc = Text_FC()
        self.text_fc.m = None
        self.text_fc.v = None
        
        self.attention = ModelAttention()
        self.attention.m = None
        self.attention.v = None
           
        # Final classifier
        self.classifier = Classifier()
        self.classifier.m = None
        self.classifier.v = None

        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, batch_data):
        """
        Forward pass of the model.
        
        Args:
            img_features (torch.Tensor): Batch of precomputed image features (shape: [batch_size, img_input_dim]).
            text_features (torch.Tensor): Batch of precomputed text features (shape: [batch_size, text_input_dim]).
            *args: Additional arguments for attention, if needed.
        
        Returns:
            outputs (torch.Tensor): Predictions for the input batch (shape: [batch_size, num_classes]).
            multimodal_repr (torch.Tensor): Multimodal representation (optional, shape: [batch_size, d_hid]).
        """

        # Process image features
        img_repr = self.img_fc(batch_data['image_feature'])
        
        # Process text features
        text_repr = self.text_fc(batch_data['text_feature'])
        
        # Attention-based fusion
        multimodal_repr, _ = self.attention(
            torch.stack([img_repr, text_repr], dim=1)  # [batch_size, 2, d_hid]
        )

        multimodal_repr = multimodal_repr.mean(dim=1)  # Average across modalities
        
        # Classification
        outputs = self.classifier(multimodal_repr)
        labels = batch_data['label']

        cls_loss = F.cross_entropy(outputs, labels)

        return cls_loss, outputs
