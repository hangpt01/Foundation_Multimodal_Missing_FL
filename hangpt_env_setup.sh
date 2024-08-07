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