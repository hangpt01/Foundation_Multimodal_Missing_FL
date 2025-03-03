git clone https://github.com/hangpt01/Foundation_Multimodal_Missing_FL.git
conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install kaggle
kaggle datasets download -d gianmarco96/upmcfood101
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html



# IMDB through ggdrive
mkdir benchmark/RAW_DATA/IMDB/generate_arrows
cd benchmark/RAW_DATA/IMDB/generate_arrows
gdown 1CWMulgXrEXKjjAoW-so2JSYRsKLKp6za
gdown 1tP11TWnlQK8x3moG2IcC8CwW33deAwLL


conda activate fmfl
pip install protobuf fonttools imgaug opencv-python pyyaml regex scipy
mkdir benchmark/RAW_DATA/FOOD101
mv upmcfood101.zip benchmark/RAW_DATA/FOOD101
cd benchmark/RAW_DATA/FOOD101
unzip upmcfood101.zip
# 3 files
pip install gdown==4.5.4 --no-cache-dir 
gdown 1YF2IbYTfYOb9ehln_EwiLZPKS6OSajya
gdown 15fZdhTr7mLsM6JIKCu74dn6-zwSZbbYF
gdown 1g-4BOS4vIdjcRLdJEHnhWPRXI0Q2JWNO
cd ../../..
pip install -r requirements_fmfl.txt
python notebook/make_arrow_food101.py
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/


# IMDB if downloaded from ggdrive
mkdir benchmark/RAW_DATA/IMDB/missing_tables/
mkdir benchmark/RAW_DATA/IMDB/missing_tables_other_tests/

# Food101
bash script/food101/8_classes/image_1_text_0.3/iid/20clients.sh

# IMDB
bash script/imdb/iid/20clients.sh



bash script/food101/8_classes/image_1_text_0.3/iid/20clients.sh
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
git pull
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
bash script/food101/8_classes/image_1_text_0.3/noniid_0.1/20cl_missing_aware.sh

