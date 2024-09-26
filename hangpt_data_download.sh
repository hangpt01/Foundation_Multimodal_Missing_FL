# move to the folder outside code repo
conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install kaggle
kaggle datasets download -d gianmarco96/upmcfood101
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# MM-IMDB


# MM-IMDB - multi-label: not choosing
mkdir benchmark/RAW_DATA/IMDB
cd benchmark/RAW_DATA/IMDB
wget https://archive.org/download/mmimdb/mmimdb.tar.gz


# Hateful memes
mkdir benchmark/RAW_DATA/HATEMEME
cd benchmark/RAW_DATA/HATEMEME
kaggle datasets download -d parthplc/facebook-hateful-meme-dataset
unzip facebook-hateful-meme-dataset.zip
gdown --id 13FdAifAZB1LgZA9Zyb4Onz7IQOTl_Ygt
rm data/test.jsonl
rm data/LICENSE.txt
rm data/README.md
mv test_seen.jsonl data/test.jsonl