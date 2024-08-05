# move to the folder outside code repo
conda create -n fmfl python=3.8 -y
conda activate fmfl
pip install kaggle
kaggle datasets download -d gianmarco96/upmcfood101
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html