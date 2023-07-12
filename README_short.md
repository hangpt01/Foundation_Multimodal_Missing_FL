
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


Run every 2 files using 1 GPU:
```sh
bash script/ptbxl_classification/new_gaga/training_case1/fedmsplit_gaga_c1_contrastive2_3_64_cnum20_dist0_skew0_seed0.sh


bash script/ptbxl_classification/new_gaga/training_case2/fedmsplit_gaga_c1_contrastive2_3_64_cnum20_dist0_skew0_seed0.sh


bash script/ptbxl_classification/new_gaga/training_case4/fedmsplit_gaga_c3_contrastive2_3_64_cnum20_dist0_skew0_seed0.sh 


```

