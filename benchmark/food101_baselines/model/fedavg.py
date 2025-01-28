import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule


class ImageTextClassifier(FModule):
    def __init__(self, d_hid, en_att=False, att_name='multihead'):
        """
        Args:
            num_classes (int): Number of output classes for classification.
            img_input_dim (int): Dimension of precomputed image features (e.g., MobileNetV2 output size).
            text_input_dim (int): Dimension of precomputed text features (e.g., MobileBERT output size).
            d_hid (int): Hidden size of intermediate layers.
            en_att (bool): Whether to use attention-based fusion.
            att_name (str): Type of attention mechanism ('multihead' or other).
        """
        super(ImageTextClassifier, self).__init__()
        num_classes = 8
        img_input_dim = 1280
        text_input_dim = 768
        d_hid = 64


        # Image feature processing
        self.img_fc = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Text feature processing
        self.text_fc = nn.Sequential(
            nn.Linear(text_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        
        self.attention = nn.MultiheadAttention(embed_dim=d_hid, num_heads=4, batch_first=True)
           
        
        # Fusion layer (concatenation by default)
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_hid * 2, d_hid),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(d_hid, num_classes)

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
        if self.en_att:
            multimodal_repr, _ = self.attention(
                torch.stack([img_repr, text_repr], dim=1),  # [batch_size, 2, d_hid]
                torch.stack([img_repr, text_repr], dim=1),
                torch.stack([img_repr, text_repr], dim=1)
            )
            multimodal_repr = multimodal_repr.mean(dim=1)  # Average across modalities
        else:
            # Concatenate features (default)
            multimodal_repr = torch.cat([img_repr, text_repr], dim=-1)
            multimodal_repr = self.fusion_fc(multimodal_repr)
        
        # Classification
        outputs = self.classifier(multimodal_repr)
        labels = batch_data['label']

        cls_loss = F.cross_entropy(outputs, labels)

        return cls_loss, outputs
