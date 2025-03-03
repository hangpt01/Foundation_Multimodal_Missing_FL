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
        self.label2idx = LABEL2IDX
        
        self.image_dir = os.path.join(self.root, 'images', self.mode)
        text_dir = os.path.join(self.root, 'texts')
        for f in os.listdir(text_dir):
            if (self.mode+'_titles') in f and os.path.isfile(os.path.join(text_dir, f)):
                text_file = os.path.join(text_dir, f)

        self.text_labels = pd.read_csv(text_file, header=None)
        self.text_labels.columns = ['image', 'caption', 'label']
        
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        print(len(set(self.text_labels['label'].tolist())))
        
    def preprocess_text(self, sen):

        def remove_tags(text):
            TAG_RE = re.compile(r'<[^>]+>')
            return TAG_RE.sub('', text)
        
        # Removing html tags
        sentence = remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        sentence = sentence.lower()

        return sentence

    def __len__(self):
        return self.text_labels.shape[0]
    
    def __getitem__(self, idx):
        text_label = self.text_labels.iloc[idx]
        im_name, caption, label = text_label['image'], text_label['caption'], text_label['label']
        img_path = os.path.join(self.image_dir, label, im_name)
        image = Image.open(img_path)
        text = self.preprocess_text(caption)
        input = self.processor(image, text, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
        # input['labels'] = torch.tensor(self.label2idx[label]).type(torch.LongTensor)
        label = torch.tensor(self.label2idx[label]).type(torch.LongTensor)
        
        # import pdb; pdb.set_trace()
        
        return input, label

LABEL2IDX = {
    "apple_pie": 0,
    "baby_back_ribs": 1,
    "baklava": 2,
    "beef_carpaccio": 3,
    "beef_tartare": 4,
    "beet_salad": 5,
    "beignets": 6,
    "bibimbap": 7,
    "bread_pudding": 8,
    "breakfast_burrito": 9,
    "bruschetta": 10,
    "caesar_salad": 11,
    "cannoli": 12,
    "caprese_salad": 13,
    "carrot_cake": 14,
    "ceviche": 15,
    "cheesecake": 16,
    "cheese_plate": 17,
    "chicken_curry": 18,
    "chicken_quesadilla": 19,
    "chicken_wings": 20,
    "chocolate_cake": 21,
    "chocolate_mousse": 22,
    "churros": 23,
    "clam_chowder": 24,
    "club_sandwich": 25,
    "crab_cakes": 26,
    "creme_brulee": 27,
    "croque_madame": 28,
    "cup_cakes": 29,
    "deviled_eggs": 30,
    "donuts": 31,
    "dumplings": 32,
    "edamame": 33,
    "eggs_benedict": 34,
    "escargots": 35,
    "falafel": 36,
    "filet_mignon": 37,
    "fish_and_chips": 38,
    "foie_gras": 39,
    "french_fries": 40,
    "french_onion_soup": 41,
    "french_toast": 42,
    "fried_calamari": 43,
    "fried_rice": 44,
    "frozen_yogurt": 45,
    "garlic_bread": 46,
    "gnocchi": 47,
    "greek_salad": 48,
    "grilled_cheese_sandwich": 49,
    "grilled_salmon": 50,
    "guacamole": 51,
    "gyoza": 52,
    "hamburger": 53,
    "hot_and_sour_soup": 54,
    "hot_dog": 55,
    "huevos_rancheros": 56,
    "hummus": 57,
    "ice_cream": 58,
    "lasagna": 59,
    "lobster_bisque": 60,
    "lobster_roll_sandwich": 61,
    "macaroni_and_cheese": 62,
    "macarons": 63,
    "miso_soup": 64,
    "mussels": 65,
    "nachos": 66,
    "omelette": 67,
    "onion_rings": 68,
    "oysters": 69,
    "pad_thai": 70,
    "paella": 71,
    "pancakes": 72,
    "panna_cotta": 73,
    "peking_duck": 74,
    "pho": 75,
    "pizza": 76,
    "pork_chop": 77,
    "poutine": 78,
    "prime_rib": 79,
    "pulled_pork_sandwich": 80,
    "ramen": 81,
    "ravioli": 82,
    "red_velvet_cake": 83,
    "risotto": 84,
    "samosa": 85,
    "sashimi": 86,
    "scallops": 87,
    "seaweed_salad": 88,
    "shrimp_and_grits": 89,
    "spaghetti_bolognese": 90,
    "spaghetti_carbonara": 91,
    "spring_rolls": 92,
    "steak": 93,
    "strawberry_shortcake": 94,
    "sushi": 95,
    "tacos": 96,
    "takoyaki": 97,
    "tiramisu": 98,
    "tuna_tartare": 99,
    "waffles": 100
}

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
