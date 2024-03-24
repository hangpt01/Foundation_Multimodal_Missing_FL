import torch
from transformers import ViltProcessor
from PIL import Image
import numpy as np
import pandas as pd
# import pickle 
import os
import re
from datetime import datetime

# Define your dataset class
def preprocess_text(sen):

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


for mode in ["train", "test"]:
    file = f'benchmark/RAW_DATA/FOOD101/texts/{mode}_titles.csv'        # 67972
    img_dir = f'benchmark/RAW_DATA/FOOD101/images/{mode}/'
    df = pd.read_csv(file, header=None)
    print(len(df))
    # import pdb; pdb.set_trace()
    df.columns = ['image', 'caption', 'label']
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    input_dict = []
    label_dict = []
    for idx in range(int(len(df))):
        text_label = df.iloc[idx]
        im_name, caption, label = text_label['image'], text_label['caption'], text_label['label']
        img_path = os.path.join(img_dir, label, im_name)
        image = Image.open(img_path)
        text = preprocess_text(caption)
        input = processor(image, text, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
        label = LABEL2IDX[label]
        
        input_dict.append(input)
        label_dict.append(label)
        if idx % 1000 == 0:
            print(datetime.now(),idx)
    
    folder_path = 'benchmark/RAW_DATA/FOOD101/101_classes'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(input_dict, os.path.join(folder_path,f'{mode}_inputs.pt'))
    torch.save(label_dict, os.path.join(folder_path,f'{mode}_labels.pt'))
    print("Saved", mode)