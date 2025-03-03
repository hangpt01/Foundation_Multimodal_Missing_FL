import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextClassifier(nn.Module):
    def __init__(self, num_classes, img_input_dim, text_input_dim, d_hid, en_att=True, att_name='multihead'):
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
        
        # Attention mechanism for multimodal fusion
        self.en_att = en_att
        if en_att:
            if att_name == 'multihead':
                self.attention = nn.MultiheadAttention(embed_dim=d_hid, num_heads=4, batch_first=True)
            else:
                raise ValueError("Unsupported attention type")
        
        # Fusion layer (concatenation by default)
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_hid * 2, d_hid),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(d_hid, num_classes)
        
    def forward(self, img_features, text_features, *args):
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
        img_repr = self.img_fc(img_features)
        
        # Process text features
        text_repr = self.text_fc(text_features)
        
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
        return outputs, multimodal_repr

# Model initialization
model = ImageTextClassifier(
    num_classes=8,           # Example: 5 classes
    img_input_dim=1280,      # Example: MobileNetV2 output
    text_input_dim=512,      # Example: MobileBERT output
    d_hid=256,               # Hidden layer size
    en_att=True,             # Enable attention-based fusion
    att_name='multihead'     # Use multihead attention
)

import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, img_features, text_features, labels):
        """
        Args:
            img_features (torch.Tensor): Precomputed image features.
            text_features (torch.Tensor): Precomputed text features.
            labels (torch.Tensor): Labels corresponding to the features.
        """
        self.img_features = img_features
        self.text_features = text_features
        self.labels = labels

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetch the features and labels for a given index.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            img_features (torch.Tensor): Image features for the sample.
            text_features (torch.Tensor): Text features for the sample.
            label (torch.Tensor): Label for the sample.
        """
        img_feature = self.img_features[idx]
        text_feature = self.text_features[idx]
        label = self.labels[idx]
        return img_feature, text_feature, label


# Load precomputed features
image_features_path = "precomputed_features/image_features_food101_test_missing_both_07_05.pt"
text_features_path = "precomputed_features/text_features_food101_test_missing_both_07_05.pt"
labels_path = "precomputed_features/labels_food101_test_missing_both_07_05.pt"

img_features = torch.load(image_features_path)
text_features = torch.load(text_features_path)
labels = torch.load(labels_path)

# Create the dataset
dataset = MultimodalDataset(img_features, text_features, labels)

# Create the dataloader
batch_size = 32  # You can adjust the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for img_batch, text_batch, label_batch in dataloader:
        # Move data to device
        img_batch = img_batch.to(device)
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        model = model.to(device)

        # Forward pass
        outputs, _ = model(img_batch, text_batch)
        
        # Compute loss
        loss = criterion(outputs, label_batch)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label_batch).sum().item()
        total += label_batch.size(0)
    
    # Print epoch statistics
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
