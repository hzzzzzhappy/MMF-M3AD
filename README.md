# MMF-M3AD
## 1. Licence
Copyright (c) 2025 Hanzhe Liang

All rights shall be reserved until the paper is accepted. This work has been submitted to Elsevier.

## 2. Quick Start

### 2.1 Requirements
```bash
conda create -n MMF-M3AD python=3.8
conda activate MMF-M3AD
pip install -r requirements.txt
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### 2.2 Pre-trained Weights
Download Point-MAE pre-trained weights from [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_8k.pth) and place the `modelnet_8k.pth` file in the `./pretrain_ckp` directory.

### 2.3 Real3D-AD

Download the dataset from [here](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing) and unzip it.

Downsample the training set:
```bash
python downsample_pcd.py --radl3d_path <Path/to/your/Real3D-AD-PCD>
```

Set `dataset.data_dir` and `net.data_dir` in `./experiments/real3d/config.yaml` to your Real3D-AD-PCD path.

Training/Evaluation:
```bash
cd ./experiments/real3d/
sh train.sh 1 0 # or sh eval.sh 1 0
```

### 2.4 Anomaly-ShapeNet

Download the dataset from [here](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet) and organize as:
```
Anomaly-ShapeNet
├── ashtray0
│   ├── train/*.pcd
│   ├── test/*.pcd
│   └── GT/*.txt
├── bag0
...
```

Set dataset paths in `./experiments/Anomaly-ShapeNet/config.yaml`.

Training/Evaluation:
```bash
cd ./experiments/Anomaly-ShapeNet/
sh train.sh 1 0 # or sh eval.sh 1 0
```

Update: We shared our [checkpoints](https://drive.google.com/file/d/10JYHvtNu3tnKm30BAGr0_uIJ6vm6_Njz/view?usp=sharing) and [visualization](https://drive.google.com/file/d/1ahqFnfAAhvkbVVJUfBRl75KqXQYPT_30/view?usp=sharing).

### 2.5 MulSen-AD

Download the dataset from [here](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main) and process following [this guide](https://github.com/hzzzzzhappy/Processing-tools-for-the-MulSen_AD-dataset.git).

Set dataset paths in `./experiments/MulSen-AD/config.yaml`.

Training/Evaluation:
```bash
cd ./experiments/MulSen-AD/
sh train.sh 1 O # or sh eval.sh 1 0
```
Note: Multi-GPU training is not supported for evaluation, set `saver.load_path` in config.yaml.
