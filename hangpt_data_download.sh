# move to the folder outside code repo
conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install kaggle
kaggle datasets download -d gianmarco96/upmcfood101
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# MM-IMDB


# MM-IMDB - multi-label: not choosing
mkdir benchmark/RAW_DATA/IMDB
cd benchmark/RAW_DATA/IMDB
wget https://archive.org/download/mmimdb/mmimdb.tar.gz


# pytorch-lightning
pip install pytorch-lightning==1.1.4
# or
pip install pytorch-lightning==1.1.4
python3 -m pip install pytorch-lightning==1.1.4


# pretrained ViLT
mkdir benchmark/pretrained_model_weight/
cd benchmark/pretrained_model_weight/
wget https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt


# # Hateful memes
# mkdir benchmark/RAW_DATA/HATEMEME
# cd benchmark/RAW_DATA/HATEMEME
# kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
# unzip facebook-hateful-meme-dataset.zip
# gdown --id 13FdAifAZB1LgZA9Zyb4Onz7IQOTl_Ygt
# rm data/test.jsonl
# rm data/LICENSE.txt
# rm data/README.md
# mv test_seen.jsonl data/test.jsonl