from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import io
import torch
import pyarrow as pa
import random
from PIL import Image
import pandas as pd
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)

from torchvision import transforms

# from .model.clip import tokenize, _transform, load
class Food101Dataset(Dataset):
    def __init__(self, root, download=True, train=True):
        super(Food101Dataset, self).__init__()
        self.root = root
        if not os.path.exists(self.root):
            if download:
                print("Download dataset ...")
                os.makedirs(root, exist_ok=True)
                os.system('bash ./benchmark/food101_classification/download.sh')
                print('done!')
        self.train = train 
        self.mode = 'train' if train else 'test'
        
        # arrow file
        tables = [
                pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{self.root}/food101_{self.mode}.arrow", "r")
            ).read_all()
        ]
        
        remove_duplicate = False
        self.table = pa.concat_tables(tables, promote=True)
        
        #-------------------------------------------------------
        # use a subset of data 
        total_rows = self.table.num_rows
        # Determine the range of rows you want to extract (for example, the first quarter of the data)
        start_index = 0
        end_index = total_rows // 30
        # Extract the subset of the table
        self.table = self.table.slice(start_index, end_index)
        #-------------------------------------------------------
        
        self.all_texts = self.table['text'].to_pandas().tolist()
        self.all_texts = (      # len: 61227
            [list(set(texts)) for texts in self.all_texts]
            if remove_duplicate
            else self.all_texts
        )
        # import pdb; pdb.set_trace()
        self.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')
        self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        
        self.transforms = [pixelbert_transform(size=384)]
        # import pdb; pdb.set_trace()
        self.max_text_len = 40
        self.index_mapper = dict()
        j = 0
        for i, texts in enumerate(self.all_texts):
            for _j in range(len(texts)):
                self.index_mapper[j] = (i, _j)
                j += 1
        # import pdb; pdb.set_trace()
        
        
    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        # import pdb; pdb.set_trace()
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_all_labels(self):
        # import pdb; pdb.set_trace()
        return self.table['label'].to_numpy()
    
    def __len__(self):
        # return self.text_labels.shape[0]
        return len(self.index_mapper)
    
    
    def __getitem__(self, index):
        # print(index)
        if isinstance(index, torch.Tensor):
            index = index.item()
        image_index, question_index = self.index_mapper[index]
        
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        # simulate_missing_type = 0
        # if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
        #     simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]
        
        # missing image, dummy image is all-one image
        # if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
        #     for idx in range(len(image_tensor)):
        #         image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        # #missing text, dummy text is ''
        # if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
        #     text = ''
        #     encoding = self.tokenizer(
        #         text,
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.max_text_len,
        #         return_special_tokens_mask=True,
        #     )   
        #     text = (text, encoding)
        # else:
        #     text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        # import pdb; pdb.set_trace()
        return {
            "image": image_tensor,
            "text": text,
            "label": labels
            # "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        }
    
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = self.collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                # import pdb; pdb.set_trace()
                
        return dict_batch

# def keys_to_transforms(keys: list, size=224):
#         return [pixelbert_transform(size=size) for key in keys]
def pixelbert_transform(size=224):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    
class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)

inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

if __name__=='__main__':
    my_dataset = Food101Dataset(root='./benchmark/RAW_DATA/FOOD101')
    my_dataset.get_all_labels()
    # my_dataset = Food101Dataset(root='./benchmark/RAW_DATA/FOOD101', train=False)
    data_loader = DataLoader(my_dataset, batch_size=1, shuffle=True, collate_fn=my_dataset.collate) # len: 61127 - batch1; 1529 0 batch 40
                                                                                                    # test:22716                    
# Iterate over the dataset and print a sample
    for batch in data_loader:
        # sample = batch[0]  # Assuming batch size is 1
        print(batch)
        import pdb; pdb.set_trace()
        
    # dataset_length = len(my_dataset)
    # one_fourth_length = dataset_length // 4

    # # Create indices for one fourth of the dataset
    # indices = torch.randperm(dataset_length)[:one_fourth_length]
    # # import pdb; pdb.set_trace()

    # # Create a Subset of the dataset with one fourth of the data
    # subset_dataset = Subset(my_dataset, indices)

    # # Create a DataLoader for the subset dataset if needed
    # subset_data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=True, collate_fn=my_dataset.collate)
    # for batch in subset_data_loader:
    #     # sample = batch[0]  # Assuming batch size is 1 - len 15281
    #     print(batch)
    #     import pdb; pdb.set_trace()
        