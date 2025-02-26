import functools
from base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import random, os
from datetime import datetime
from transformers import BertTokenizer, DataCollatorForLanguageModeling

class FOOD101Dataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "test"]
        self.split = split

        if split == "train":
            names = ["food101_train"]
        elif split == "test":
            names = ["food101_test"] 

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.','')
        missing_type = missing_info['type'][split]    
        both_ratio = missing_info['both_ratio']
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.table['image'])
        
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1
                    missing_index_image  = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table


    def __getitem__(self, index):
        # index -> pair data index
        # image_index -> image index in table
        # question_index -> plot index in texts of the given image
        image_index, question_index = self.index_mapper[index]
        
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        simulate_missing_type = 0
        if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        
        # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        #missing text, dummy text is ''
        if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
            text = ''
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )   
            text = (text, encoding)
        else:
            text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        # import pdb; pdb.set_trace()
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        }

if __name__=='__main__':
    print(datetime.now(), "Start creating Datasets")
    data_dir = "../../benchmark/RAW_DATA/FOOD101/generate_arrows"
    transform_keys = ['pixelbert']
    split="train"
    image_size = 384
    max_text_len = 40
    draw_false_image = 0
    draw_false_text = 0
    image_only = False
    _config = {
        'missing_ratio':
            {'test': 0.7,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/FOOD101/missing_tables/',
        'missing_type':
            {'test': 'both',
            'train': 'both'},
        'both_ratio': 0.5,
        'simulate_missing': False
    }
    missing_info = {
            'ratio' : _config["missing_ratio"],
            'type' : _config["missing_type"],
            'both_ratio' : _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'simulate_missing' : _config["simulate_missing"]
        }
        # for bash execution
    # if _config["test_ratio"] is not None:
    #     missing_info['ratio']['val'] = _config["test_ratio"]
    #     missing_info['ratio']['test'] = _config["test_ratio"]
    # if _config["test_type"] is not None:
    #     missing_info['type']['val'] = _config["test_type"]
    #     missing_info['type']['test'] = _config["test_type"]
            
    train_dataset = FOOD101Dataset(data_dir, transform_keys, split=split, 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info)
    train_dataset.tokenizer = BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased')

    collator = DataCollatorForLanguageModeling

    train_dataset.mlm_collator = collator(tokenizer=train_dataset.tokenizer, mlm=True, mlm_probability=0.15)
    
    train_dataset.collate = functools.partial(train_dataset.collate, mlm_collator=train_dataset.mlm_collator)


    batch_size = 32
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=train_dataset.collate)
    batch = next(iter(train_dataloader))
    # img, text, label, missing_type = batch['image'], batch['text'], batch['label'], batch['missing_type']
    import pdb; pdb.set_trace()

    # print(label.shape, label)
    # print(missing_type)
    # for k,v in img.items():
    #     print("Image sample", k, v.shape)
    # for k,v in text.items():
    #     print("Image sample", k, v.shape)

    # import pdb; pdb.set_trace()