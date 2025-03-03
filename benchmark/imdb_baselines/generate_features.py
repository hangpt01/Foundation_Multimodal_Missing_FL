import functools
# from .base_dataset import BaseDataset
from base_dataset import BaseDataset        # run __main__
from torch.utils.data import DataLoader
import torch
import random, os
from datetime import datetime
# from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import MobileBertTokenizer, MobileBertModel, DataCollatorForLanguageModeling
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from collections import Counter
import sys

class IMDBDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, feature_dir="./precomputed_features", **kwargs):
        # assert split in ["train", "test"]
        self.split = split
        self.feature_dir = feature_dir

        if split == "train":
            names = ["mmimdb_train"]
        else:
            names = ["mmimdb_test"] 
        # import pdb; pdb.set_trace()
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="plots",
            split=self.split,
            remove_duplicate=False
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        # import pdb; pdb.set_trace()
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.','')
        missing_type = missing_info['type'][split]       
        both_ratio = missing_info['both_ratio']         # 0.5
        bratio = str(both_ratio).replace('.','')
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}_{bratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.table['image'])
        # import pdb; pdb.set_trace()
        
        if os.path.exists(missing_table_path):
            # import pdb; pdb.set_trace()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.image_model = mobilenet_v2(pretrained=True).to(self.device)
        self.image_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(self.device)

        self.image_model.classifier = torch.nn.Identity()  # Remove the classification head
        self.image_model.eval()

        self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        self.text_model = MobileBertModel.from_pretrained("google/mobilebert-uncased").to(self.device)
        self.text_model.eval()

        # Generate or load precomputed features
        feature_suffix = f"{names[0]}_missing_{missing_type}_{mratio}_{bratio}.pt"
        self.precomputed_image_features_path = os.path.join(self.feature_dir, f"image_features_{feature_suffix}")
        self.precomputed_text_features_path = os.path.join(self.feature_dir, f"text_features_{feature_suffix}")
        self.precomputed_labels_path = os.path.join(self.feature_dir, f"labels_{feature_suffix}")


        self._generate_or_load_features()

    def _generate_or_load_features(self):
        """Generate or load precomputed features."""
        if (
            os.path.exists(self.precomputed_image_features_path)
            and os.path.exists(self.precomputed_text_features_path)
            and os.path.exists(self.precomputed_labels_path)
        ):
            print(f"Loading precomputed features for {self.split} split...")
            self.image_features = torch.load(self.precomputed_image_features_path)
            self.text_features = torch.load(self.precomputed_text_features_path)
            self.labels = torch.load(self.precomputed_labels_path)
        else:
            print(f"Generating precomputed features for {self.split} split...")
            # import pdb; pdb.set_trace()

            image_features_list = []
            text_features_list = []
            labels_list = []

            loader = DataLoader(self, batch_size=512, shuffle=False, num_workers=8)
            with torch.no_grad():
                for batch in loader:
                    # Image feature extraction
                    images = batch['image'][0].to(self.device)
                    image_features = self.image_model(images)
                    image_features_list.append(image_features.cpu())

                    # Text feature extraction
                    input_ids = torch.stack([text for text in batch['text'][1]['input_ids']], axis=1).to(self.device)
                    attention_mask = torch.stack([text for text in batch['text'][1]['attention_mask']], axis=1).to(self.device)
                    text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                    text_features = text_outputs.pooler_output
                    text_features_list.append(text_features.cpu())

                    # Collect labels
                    labels_list.extend(batch["label"])

            # Save precomputed features
            self.image_features = torch.cat(image_features_list, dim=0)
            self.text_features = torch.cat(text_features_list, dim=0)
            self.labels = torch.tensor(labels_list)

            os.makedirs(self.feature_dir, exist_ok=True)
            torch.save(self.image_features, self.precomputed_image_features_path)
            torch.save(self.text_features, self.precomputed_text_features_path)
            torch.save(self.labels, self.precomputed_labels_path)
            # import pdb; pdb.set_trace()


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
            # import pdb; pdb.set_trace()
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
    data_dir = "../../benchmark/RAW_DATA/IMDB/generate_arrows"
    feature_dir = "./precomputed_features"
    transform_keys = ['pixelbert']
    image_size = 224
    max_text_len = 40
    draw_false_image = 0
    draw_false_text = 0
    image_only = False
    _config = {
        'missing_ratio':
            {'test': 0.7,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
        'missing_type':
            {'test': 'text',
            'train': 'text'},
        'both_ratio': 0,
        'simulate_missing': False
    }
    missing_info = {
            'ratio' : _config["missing_ratio"],
            'type' : _config["missing_type"],
            'both_ratio' : _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'simulate_missing' : _config["simulate_missing"]
        }
            
    train_dataset = IMDBDataset(data_dir, transform_keys, split='train', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)
    test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)
    _config = {
        'missing_ratio':
            {'test': 0.7,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
        'missing_type':
            {'test': 'image',
            'train': 'image'},
        'both_ratio': 0,
        'simulate_missing': False
    }
    missing_info = {
            'ratio' : _config["missing_ratio"],
            'type' : _config["missing_type"],
            'both_ratio' : _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'simulate_missing' : _config["simulate_missing"]
        }
    train_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)
    test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)
    # _config = {
    #     'missing_ratio':
    #         {'test': 0.7,
    #         'train': 0.7},
    #     'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
    #     'missing_type':
    #         {'test': 'both',
    #         'train': 'both'},
    #     'both_ratio': 0.5,
    #     'simulate_missing': False
    # }
    # missing_info = {
    #         'ratio' : _config["missing_ratio"],
    #         'type' : _config["missing_type"],
    #         'both_ratio' : _config["both_ratio"],
    #         'missing_table_root': _config["missing_table_root"],
    #         'simulate_missing' : _config["simulate_missing"]
    #     }
            
    # train_dataset = IMDBDataset(data_dir, transform_keys, split='train', 
    #                             image_size=image_size,
    #                             max_text_len=max_text_len,
    #                             draw_false_image=draw_false_image,
    #                             draw_false_text=draw_false_text,
    #                             image_only=image_only,
    #                             missing_info=missing_info,
    #                             feature_dir=feature_dir)
    # test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
    #                             image_size=image_size,
    #                             max_text_len=max_text_len,
    #                             draw_false_image=draw_false_image,
    #                             draw_false_text=draw_false_text,
    #                             image_only=image_only,
    #                             missing_info=missing_info,
    #                             feature_dir=feature_dir)
    # _config = {
    #     'missing_ratio':
    #         {'test': 0,
    #         'train': 0.7},
    #     'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
    #     'missing_type':
    #         {'test': 'both',
    #         'train': 'both'},
    #     'both_ratio': 0,
    #     'simulate_missing': False
    # }
    # missing_info = {
    #         'ratio' : _config["missing_ratio"],
    #         'type' : _config["missing_type"],
    #         'both_ratio' : _config["both_ratio"],
    #         'missing_table_root': _config["missing_table_root"],
    #         'simulate_missing' : _config["simulate_missing"]
    #     }
    # test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
    #                             image_size=image_size,
    #                             max_text_len=max_text_len,
    #                             draw_false_image=draw_false_image,
    #                             draw_false_text=draw_false_text,
    #                             image_only=image_only,
    #                             missing_info=missing_info,
    #                             feature_dir=feature_dir)
    # _config = {
    #     'missing_ratio':
    #         {'test': 1,
    #         'train': 0.7},
    #     'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
    #     'missing_type':
    #         {'test': 'text',
    #         'train': 'both'},
    #     'both_ratio': 0,
    #     'simulate_missing': False
    # }
    # missing_info = {
    #         'ratio' : _config["missing_ratio"],
    #         'type' : _config["missing_type"],
    #         'both_ratio' : _config["both_ratio"],
    #         'missing_table_root': _config["missing_table_root"],
    #         'simulate_missing' : _config["simulate_missing"]
    #     }
    # test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
    #                             image_size=image_size,
    #                             max_text_len=max_text_len,
    #                             draw_false_image=draw_false_image,
    #                             draw_false_text=draw_false_text,
    #                             image_only=image_only,
    #                             missing_info=missing_info,
    #                             feature_dir=feature_dir)
    
    # _config = {
    #     'missing_ratio':
    #         {'test': 1,
    #         'train': 0.7},
    #     'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
    #     'missing_type':
    #         {'test': 'image',
    #         'train': 'both'},
    #     'both_ratio': 0,
    #     'simulate_missing': False
    # }
    # missing_info = {
    #         'ratio' : _config["missing_ratio"],
    #         'type' : _config["missing_type"],
    #         'both_ratio' : _config["both_ratio"],
    #         'missing_table_root': _config["missing_table_root"],
    #         'simulate_missing' : _config["simulate_missing"]
    #     }
    # test_dataset = IMDBDataset(data_dir, transform_keys, split='test', 
    #                             image_size=image_size,
    #                             max_text_len=max_text_len,
    #                             draw_false_image=draw_false_image,
    #                             draw_false_text=draw_false_text,
    #                             image_only=image_only,
    #                             missing_info=missing_info,
    #                             feature_dir=feature_dir)
    os._exit(1)