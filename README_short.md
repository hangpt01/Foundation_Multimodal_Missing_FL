
## Requirements

Create conda environment & install required packages: 
```sh
conda create -n "mmfl" python=3.9
conda install -c anaconda cudatoolkit
pip install -r requirements.txt
```

Log in wandb:
```sh
wandb login
```

Paste API key (first log-in time): 887703a5a8c2fde78c2813ca42da60726b8e0cda


```sh
cd cd script/ptbxl_classification/modal_equal_gaga
```

Create 8 windows to run 8 script files:
```sh
bash fedmsplit_gaga_c1_2_128_cnum20_dist0_skew0_seed0.sh 
bash fedmsplit_gaga_c3_cnum20_dist0_skew0_seed0.sh
bash fedmsplit_gaga_c1_cnum20_dist0_skew0_seed0.sh
bash fedmsplit_gaga_c3_2_128_cnum20_dist0_skew0_seed0.sh    
bash fedmsplit_gaga_cnum20_dist0_skew0_seed0.sh 
bash fedmsplit_gaga_c4_2_128_cnum20_dist0_skew0_seed0.sh  
bash fedmsplit_gaga_2_128_cnum20_dist0_skew0_seed0.sh    
bash fedmsplit_gaga_c4_cnum20_dist0_skew0_seed0.sh   
```

