import torch
from transformers import ViltProcessor
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match VILT model input size
    transforms.ToTensor(),           # Convert PIL image to PyTorch tensor
])

# Initialize VILT processor
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Define your text and image
text = "Crock-Pot Ladies  Crock-Pot Apple Pie Moonshine"
image_path = "benchmark/RAW_DATA/FOOD101/images/train/beef_carpaccio/beef_carpaccio_0.jpg"  # Provide the path to your image
image = Image.open(image_path)
# print('---------------------------------------------')
# print('Original image',transform(image))
# # Preprocess text and image
# features = vilt_processor(image, text, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
# print('---------------------------------------------')
# print('Full modal',features)
#   input_ids           [101,x,y,,,,.,z,102,0,0,...,0]
#   token_type_ids      Full 0
#   attention_mask      [1,1,1,1,1,1,1,1,1,1,0,0,...,0]
#   pixel_values
#   pixel_mask          Full 1

# import pdb; pdb.set_trace()
# no_text = vilt_processor(image, "", padding="max_length", truncation=True, max_length=40, return_tensors="pt")
# print('---------------------------------------------')
# print('Img only',no_text)
# #   input_ids           [101,102,0,0,...,0]
# #   token_type_ids      Same
# #   attention_mask      [1,1,0,0,...,0]
# #   pixel_values        Same
# #   pixel_mask          Same


# feat_to_no_text = features
# feat_to_no_text['input_ids'] = torch.zero_like(features['input_ids'])
# feat_to_no_text['input_ids']


no_img = vilt_processor(torch.ones(transform(image).shape), text, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
print('---------------------------------------------')
print('Text only',no_img)
#   input_ids             Same
#   token_type_ids        Same
#   attention_mask        Same
#   pixel_values          Set to all 1
#   pixel_mask            Same

text2 = text + "this is class pie"
no_img_2 = vilt_processor(torch.ones(transform(image).shape), text2, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
print('---------------------------------------------')
print('Text only 2',no_img_2)
'''
=>  Append prompt: change only input_ids & attention_mask
    input ids: add the additional part to 
e.g: [  101, 13675,  7432,  1011,  8962,  6456, 13675,  7432,  1011,  8962,
          6207, 11345, 23377, 14014,                                      102,...]
to [  101, 13675,  7432,  1011,  8962,  6456, 13675,  7432,  1011,  8962,
          6207, 11345, 23377, 14014, 15222,  2015,  2003,  2465, 11345,   102,...]
and: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                  ...]
to:  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   ...]

'''

import pdb; pdb.set_trace()

'''
=> keep the same: token_type_ids, pixel_mask
=>  [0]: image-only: set input_ids & attention_mask
=>  [1]: text-only:  set pixel_values to all 1

'''
