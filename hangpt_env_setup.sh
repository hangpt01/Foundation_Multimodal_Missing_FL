# move to the folder outside code repo
conda activate fmfl
pip install protobuf fonttools imgaug opencv-python pyyaml regex scipy
git clone https://github.com/hangpt01/Foundation_Multimodal_Missing_FL.git
mkdir Foundation_Multimodal_Missing_FL/benchmark/RAW_DATA/FOOD101
mv upmcfood101.zip Foundation_Multimodal_Missing_FL/benchmark/RAW_DATA/FOOD101
cd Foundation_Multimodal_Missing_FL/benchmark/RAW_DATA/FOOD101
unzip upmcfood101.zip
# 3 files
pip install gdown==4.5.4 --no-cache-dir 
gdown 1YF2IbYTfYOb9ehln_EwiLZPKS6OSajya
gdown 15fZdhTr7mLsM6JIKCu74dn6-zwSZbbYF
gdown 1g-4BOS4vIdjcRLdJEHnhWPRXI0Q2JWNO
cd ../../..
pip install -r requirements.txt
python notebook/make_arrow.py
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/
bash script/food101/8_classes/image_1_text_0.3/iid/20clients.sh
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
git pull
bash script/food101/8_classes/image_1_text_0.3/iid/20cl_missing_aware.sh
bash script/food101/8_classes/image_1_text_0.3/noniid_0.1/20cl_missing_aware.sh


# MM-IMDB
cd benchmark/RAW_DATA/IMDB
tar -xzvf mmimdb.tar.gz
cd ../../..
python notebook/create_fols_imdb.py
rm -r benchmark/RAW_DATA/IMDB/mmimdb
python notebook/make_arrow_imdb.py
mkdir benchmark/RAW_DATA/IMDB/missing_tables/
mkdir benchmark/RAW_DATA/IMDB/missing_tables_other_tests/


# change mmimdb arrow
rm -r benchmark/RAW_DATA/IMDB/generate_arrows
rm -r benchmark/RAW_DATA/IMDB/missing_tables/
rm -r benchmark/RAW_DATA/IMDB/missing_tables_other_tests/
python notebook/make_arrow_imdb.py
mkdir benchmark/RAW_DATA/IMDB/missing_tables/
mkdir benchmark/RAW_DATA/IMDB/missing_tables_other_tests/



# Or download the arrows from ggdrive
# test
gdown 1CWMulgXrEXKjjAoW-so2JSYRsKLKp6za
# train
gdown 


# HATEMEMES
python notebook/make_arrow_hatememe.py
mkdir benchmark/RAW_DATA/HATEMEME/missing_tables/
mkdir benchmark/RAW_DATA/HATEMEME/missing_tables_other_tests/
