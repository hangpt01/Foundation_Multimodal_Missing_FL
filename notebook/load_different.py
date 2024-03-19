import torch
from transformers import ViltProcessor, ViltModel, AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# import pickle 
import os
import re
from torch import nn
from datetime import datetime

# Define your dataset class


class Food101Dataset(Dataset):
    def __init__(self, root, download=False, train=True, subset=False):
        super(Food101Dataset, self).__init__()
        self.root = root
        
        # if not os.path.exists(self.root):
        #     print("Download dataset ...")
        #     os.system('bash ./benchmark/food101_classification/download.sh')
        
        # self.stop_chars = ["#","|","%","@","&",";",".com",":","\\","/",">","<","=","{","}"]
        self.train = train 
        self.mode = 'train' if train else 'test'
        self.x = torch.load(os.path.join(self.root, f'{self.mode}_inputs.pt'))       
        self.y = torch.load(os.path.join(self.root, f'{self.mode}_labels.pt'))       
        self.y = np.array(self.y)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        input = self.x[index]
        label = self.y[index]
        
        import pdb; pdb.set_trace()
        return input, label

def collate(batch):
    # import pdb; pdb.set_trace()
    batch_ = batch
    batch = []
    labels = []
    for input, label in batch_:
        batch.append(input)
        labels.append(label)
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'][0] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    # import pdb; pdb.set_trace()
    # labels = [item['labels'].item() for item in batch]

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    
    # import pdb; pdb.set_trace()
    batch = {}
    batch['input_ids'] = torch.cat(input_ids)
    batch['attention_mask'] = torch.cat(attention_mask)
    batch['token_type_ids'] = torch.cat(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    # batch['labels'] = torch.tensor(labels)
    return batch, torch.tensor(labels)   

def load_data():
    rawdata_path = 'benchmark/RAW_DATA/FOOD101'
    train_data = Food101Dataset(
        root=rawdata_path,
        download=True,
        train=True
    )
    # import pdb; pdb.set_trace()
    test_data = Food101Dataset(
        root=rawdata_path,
        download=True,
        train=False
    )
    return train_data, test_data
# Define your ViLT model (CLIP)
# model_name = "openai/clip-vit-base-patch16"
# tokenizer = CLIPProcessor.from_pretrained(model_name)
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# model = CLIPModel.from_pretrained(model_name)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
# Freeze the parameters of the ViLT model
for param in model.parameters():
    param.requires_grad = False
# classifier = torch.nn.Linear(768, 2)
classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 101),
)

# Define your dataset and split it into train and validation sets
# texts, images, labels = load_data()
# # import pdb; pdb.set_trace()

# train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(texts, images, labels, test_size=0.2)

# Initialize datasets and data loaders
train_dataset, val_dataset = load_data()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=collate)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
classifier.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(datetime.now(),'Start epoch', epoch+1)
    i=1
    for batch, labels in train_loader:
        print(datetime.now(),'Load batch', i)
        batch = {k:v.to(device) for k,v in batch.items()}
        print(datetime.now(),'Start model')
        with torch.no_grad():
            outputs = model(**batch)
        features = outputs.last_hidden_state[:, 0, :]
        predictions = classifier(features) 
        loss = criterion(predictions, labels.to(device))
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        print(datetime.now(),'End batch', i)
        i += 1
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch, labels in val_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            features = outputs.last_hidden_state[:, 0, :]
            predictions = classifier(features) 
            val_loss = criterion(predictions, labels.to(device))
            val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_models/small_model")
