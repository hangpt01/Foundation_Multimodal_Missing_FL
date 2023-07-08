
## Requirements

Create conda environment & install required packages: 
```sh
conda create -n "mmfl" python=3.9
conda activate mmfl
conda install -c anaconda cudatoolkit
pip install -r requirements.txt
```

Log in wandb:
```sh
wandb login
```

Paste API key (first log-in time): 887703a5a8c2fde78c2813ca42da60726b8e0cda


Run every 2 files using 1 GPU:
```sh
bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_3_64_cnum20_dist0_skew0_seed0.sh
bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_gaga_c1_5_64_cnum20_dist0_skew0_seed0.sh 


bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_5_64_cnum20_dist0_skew0_seed0.sh
bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_gaga_c1_3_64_cnum20_dist0_skew0_seed0.sh  


bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_contrastive5_3_64_cnum20_dist0_skew0_seed0
bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_gaga_c3_5_64_cnum20_dist0_skew0_seed0.sh  


bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_contrastive5_5_64_cnum20_dist0_skew0_seed0    
bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_gaga_c3_3_64_cnum20_dist0_skew0_seed0.sh   
```

