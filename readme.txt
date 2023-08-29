# create enviroment
conda create -n codef python=3.10

# install pytorch 2.0.0
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117


# install requre enviroment
sudo apt-get install ffmpeg

pip install -r requirements.txt


# install tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

验证  import tinycudann as tcnn


# Train a new model
./scripts/train_multi.sh


# Test reconstruction
./scripts/test_multi.sh


# 视频转换
./scripts/test_canonical.sh


