# SDC-Radar-Detection

## Installation

We use MMRotate as our detection frameworkm, which requires Python 3.7+, CUDA 9.2+ and PyTorch 1.6+.

```bash
git clone https://github.com/jimmylin0979/SDC-Radar-Detection.git
cd SDC-Radar-Detection

# pytorch
# please install following https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# mmrotate
# please install mmrotate following https://mmrotate.readthedocs.io/en/latest/install.html#installation
cd mmroate 
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
pip install -v -e .

# install other dependencies
cd ..
pip install -r requirements.txt
```

## Training

```bash
# cd in mmrotate folder
cd mmrotate

# the training command in mmrotate
# train model declared in CONFIG_FILE, and store everything in WORK_DIR
CONFIG_FILE=???
WORK_DIR=???
python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR}

# Example
# train with redet with config redet_re50_refpn_1x_dota_ms_rr_le90, and store the checkpoint into work dir
python3 tools/train.py configs/sdc/redet_re50_refpn_1x_dota_ms_rr_le90.py --work-dir ../results/redet_re50_refpn_1x_dota_ms_rr_le90
```

## Inference

```bash
# inference the rotated object detection model with specific config files, checkpoint path and image root
# it will generate a viz folder for visualization and a json file for predictions
CONFIG_FILE=???
CKPT_DIR=???
IMG_DIR=???
python inference.py --config ${CONFIG_FILE} --ckpt ${CKPT_DIR} --root ${IMG_DIR}

# Example
# inference the redet with config, and checkpoint on specific image root 
python inference.py --config results/redet_re50_refpn_1x_dota_ms_rr_le90_batch2/redet_re50_refpn_1x_dota_ms_rr_le90.py --ckpt results/redet_re50_refpn_1x_dota_ms_rr_le90_batch2/latest.pth --root ./data/mini_test/city_7_0/Navtech_Cartesia
```