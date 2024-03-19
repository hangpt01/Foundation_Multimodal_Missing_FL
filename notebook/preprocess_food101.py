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
    for idx in range(int(len(df)/33.4)):
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
    torch.save(input_dict, f'benchmark/RAW_DATA/FOOD101/{mode}_inputs.pt')
    torch.save(label_dict, f'benchmark/RAW_DATA/FOOD101/{mode}_labels.pt')
    print("Saved", mode)