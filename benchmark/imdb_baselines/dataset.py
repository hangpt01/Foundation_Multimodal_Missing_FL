import functools
from .base_dataset import BaseDataset
# from base_dataset import BaseDataset        # run __main__
from torch.utils.data import DataLoader
import torch
import random, os
from datetime import datetime
# from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import MobileBertTokenizer, MobileBertModel, DataCollatorForLanguageModeling
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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
            remove_duplicate=False,
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

        self.image_features = torch.load(self.precomputed_image_features_path)
        self.text_features = torch.load(self.precomputed_text_features_path)
        self.labels = torch.load(self.precomputed_labels_path)

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        # Retrieve the precomputed features
        image_feature = self.image_features[index]
        text_feature = self.text_features[index]
        label = self.labels[index]

        # Handle missing modalities
        image_index, question_index = self.index_mapper[index]  # Access the original table index
        simulate_missing_type = 0

        if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0, 1, 2])  # Randomly simulate missing modality during training

        # Handle missing image modality
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            image_feature = torch.zeros_like(image_feature)  # Replace image feature with zeros

        # Handle missing text modality
        if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
            text_feature = torch.zeros_like(text_feature)  # Replace text feature with zeros

        return {
            "image_feature": image_feature,
            "text_feature": text_feature,
            "label": label,
            "missing_type": self.missing_table[image_index].item() + simulate_missing_type,
        }
        

if __name__=='__main__':
    print(datetime.now(), "Start creating Datasets")
    data_dir = "../../benchmark/RAW_DATA/IMDB/generate_arrows"
    feature_dir = "./precomputed_features"
    transform_keys = ['pixelbert']
    split="test"
    image_size = 224
    max_text_len = 40
    draw_false_image = 0
    draw_false_text = 0
    image_only = False
    _config = {
        'missing_ratio':
            {'test': 1,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/IMDB/missing_tables_other_tests/',
        'missing_type':
            {'test': 'text',
            'train': 'both'},
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
        # for bash execution
    # if _config["test_ratio"] is not None:
    #     missing_info['ratio']['val'] = _config["test_ratio"]
    #     missing_info['ratio']['test'] = _config["test_ratio"]
    # if _config["test_type"] is not None:
    #     missing_info['type']['val'] = _config["test_type"]
    #     missing_info['type']['test'] = _config["test_type"]
            
    train_dataset = IMDBDataset(data_dir, transform_keys, split=split, 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)

    batch_size = 32
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    batch = next(iter(train_dataloader))
    
    # img, text, label, missing_type = batch['image'], batch['text'], batch['label'], batch['missing_type']
    # import pdb; pdb.set_trace()
    os._exit(1)


    # print(label.shape, label)
    # print(missing_type)
    # for k,v in img.items():
    #     print("Image sample", k, v.shape)
    # for k,v in text.items():
    #     print("Image sample", k, v.shape)

    # import pdb; pdb.set_trace()