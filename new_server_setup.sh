git clone https://github.com/hangpt01/Foundation_Multimodal_Missing_FL.git
cd Foundation_Multimodal_Missing_FL
conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install -r requirements_fmfl.txt
pip install protobuf fonttools imgaug opencv-python pyyaml regex scipy
mkdir benchmark/pretrained_model_weight/
cd benchmark/pretrained_model_weight/
wget https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt
git clone https://huggingface.co/bert-base-uncased
cd ../..


# IMDB through ggdrive
pip install gdown==4.5.4 --no-cache-dir 
mkdir -p benchmark/RAW_DATA/IMDB/generate_arrows
cd benchmark/RAW_DATA/IMDB/generate_arrows
gdown 1CWMulgXrEXKjjAoW-so2JSYRsKLKp6za
gdown 1tP11TWnlQK8x3moG2IcC8CwW33deAwLL
# IMDB if downloaded from ggdrive
mkdir benchmark/RAW_DATA/IMDB/missing_tables/
mkdir benchmark/RAW_DATA/IMDB/missing_tables_other_tests/
# Test the set up of IMDB
bash script/imdb/miss_both/20clients.sh


# Food101
conda activate fmfl
mkdir -p benchmark/RAW_DATA/FOOD101
cd benchmark/RAW_DATA/FOOD101
curl -L -o upmcfood101.zip https://www.kaggle.com/api/v1/datasets/download/gianmarco96/upmcfood101
unzip upmcfood101.zip
# 3 files
gdown 1YF2IbYTfYOb9ehln_EwiLZPKS6OSajya
gdown 15fZdhTr7mLsM6JIKCu74dn6-zwSZbbYF
gdown 1g-4BOS4vIdjcRLdJEHnhWPRXI0Q2JWNO
cd ../../..
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/
python notebook/make_arrow_food101.py

# Food101
bash script/food101/8_classes/miss_both/20clients.sh


bash script/food101/8_classes/image_1_text_0.3/iid/20clients.sh
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
git pull
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
bash script/food101/8_classes/image_1_text_0.3/noniid_0.1/20cl_missing_aware.sh

