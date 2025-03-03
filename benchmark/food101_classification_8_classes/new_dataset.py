import functools
# from .base_dataset import BaseDataset
from base_dataset import BaseDataset        # run __main__
from torch.utils.data import DataLoader
import torch
import random, os
from datetime import datetime
# from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import MobileBertTokenizer, MobileBertModel, DataCollatorForLanguageModeling
from torchvision.models import mobilenet_v2
from collections import Counter

class FOOD101Dataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, feature_dir="./precomputed_features", **kwargs):
        # assert split in ["train", "test"]
        self.split = split
        self.feature_dir = feature_dir

        if split == "train":
            names = ["food101_train"]
        else:
            names = ["food101_test"] 
        # import pdb; pdb.set_trace()
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="text",
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
        self.image_model = mobilenet_v2(pretrained=True).to(self.device)
        self.image_model.classifier = torch.nn.Identity()  # Remove the classification head
        self.image_model.eval()

        self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        self.text_model = MobileBertModel.from_pretrained("google/mobilebert-uncased").to(self.device)
        self.text_model.eval()

        # Generate or load precomputed features
        self.precomputed_image_features_path = os.path.join(self.feature_dir, f"{split}_image_features.pt")
        self.precomputed_text_features_path = os.path.join(self.feature_dir, f"{split}_text_features.pt")
        self.precomputed_labels_path = os.path.join(self.feature_dir, f"{split}_labels.pt")

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
            image_features_list = []
            text_features_list = []
            labels_list = []

            loader = DataLoader(self, batch_size=32, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    # Image feature extraction
                    images = torch.stack([torch.stack(img) for img in batch["image"]]).to(self.device)
                    image_features = self.image_model(images)
                    image_features_list.append(image_features.cpu())

                    # Text feature extraction
                    input_ids = torch.stack([text["input_ids"] for text in batch["text"]]).to(self.device)
                    attention_mask = torch.stack([text["attention_mask"] for text in batch["text"]]).to(self.device)
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
        # return {
        #     "image": image_tensor,
        #     "text": text,
        #     "label": labels,
        #     "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        # }
        import pdb; pdb.set_trace()
        image_feature = self.image_features[index]
        text_feature = self.text_features[index]
        label = self.labels[index]

        return {
            "image_feature": image_feature,
            "text_feature": text_feature,
            "label": label,
        }

if __name__=='__main__':
    print(datetime.now(), "Start creating Datasets")
    data_dir = "../../benchmark/RAW_DATA/FOOD101/generate_arrows"
    feature_dir = "./precomputed_features"
    transform_keys = ['pixelbert']
    split="test"
    image_size = 384
    max_text_len = 40
    draw_false_image = 0
    draw_false_text = 0
    image_only = False
    _config = {
        'missing_ratio':
            {'test': 1,
            'train': 0.7},
        'missing_table_root': '../../benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/',
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
            
    train_dataset = FOOD101Dataset(data_dir, transform_keys, split=split, 
                                image_size=image_size,
                                max_text_len=max_text_len,
                                draw_false_image=draw_false_image,
                                draw_false_text=draw_false_text,
                                image_only=image_only,
                                missing_info=missing_info,
                                feature_dir=feature_dir)
    
    collator = DataCollatorForLanguageModeling

    train_dataset.mlm_collator = collator(tokenizer=train_dataset.tokenizer, mlm=True, mlm_probability=0.15)
    
    train_dataset.collate = functools.partial(train_dataset.collate, mlm_collator=train_dataset.mlm_collator)

    # import pdb; pdb.set_trace()
    missing_types = []
    labels = []
    for i in range(len(train_dataset)):
        data_sample = train_dataset[i]
        missing_type = data_sample["missing_type"]
        missing_types.append(missing_type)
        label = data_sample["label"]
        labels.append(label)

    dict_types = Counter(missing_types)
    dict_labels = Counter(labels)
    str_ = '\t' + str({k: dict_types[k] for k in sorted(dict_types)}) + '\t\t' + str({k: dict_labels[k] for k in sorted(dict_labels)})
    print(str_)
    exit()
    # import pdb; pdb.set_trace()

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