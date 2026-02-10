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
sh train_torch.sh 1 0 # or sh eval_torch.sh 1 0
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
sh train_torch.sh 1 0 # or sh eval_torch.sh 1 0
```

Update: We shared our [checkpoints](https://drive.google.com/file/d/10JYHvtNu3tnKm30BAGr0_uIJ6vm6_Njz/view?usp=sharing) and [visualization](https://drive.google.com/file/d/1ahqFnfAAhvkbVVJUfBRl75KqXQYPT_30/view?usp=sharing).

If you use this checkpoints, you will get following results:
| clsname | obj-AUROC | pixel-AUROC |
|:---:|:---:|:---:|
| ashtray0 | 0.995238 | 0.897062 |
| ashtray0\|bulge | 1 | 0.889446 |
| ashtray0\|concavity | 0.990476 | 0.937943 |
| bag0 | 0.9 | 0.866893 |
| bag0\|bulge | 0.895238 | 0.856569 |
| bag0\|concavity | 0.904762 | 0.906479 |
| bottle0 | 0.942857 | 0.936578 |
| bottle0\|bulge | 0.914286 | 0.933266 |
| bottle0\|concavity | 0.971429 | 0.953431 |
| bottle1 | 0.842105 | 0.908467 |
| bottle1\|broken | 0.766667 | 0.305473 |
| bottle1\|bulge | 0.819048 | 0.905599 |
| bottle1\|concavity | 0.933333 | 0.948842 |
| bottle1\|crak | 0.733333 | 0.206387 |
| bottle1\|hole | 0.733333 | 0.495934 |
| bottle3 | 0.936508 | 0.932748 |
| bottle3\|bulge | 0.92381 | 0.959732 |
| bottle3\|concavity | 0.961905 | 0.964948 |
| bottle3\|crak | 1 | 0.838852 |
| bottle3\|hole | 1 | 0.9109 |
| bottle3\|scratch | 0.822222 | 0.692359 |
| bowl0 | 1 | 0.903884 |
| bowl0\|bulge | 1 | 0.916167 |
| bowl0\|concavity | 1 | 0.928045 |
| bowl0\|scratch | 1 | 0.834646 |
| bowl1 | 0.907407 | 0.648632 |
| bowl1\|bulge | 0.87619 | 0.685258 |
| bowl1\|concavity | 0.895238 | 0.702776 |
| bowl1\|scratch | 0.983333 | 0.508401 |
| bowl2 | 0.907407 | 0.75407 |
| bowl2\|bulge | 0.866667 | 0.826114 |
| bowl2\|concavity | 0.92381 | 0.827766 |
| bowl2\|scratch | 0.95 | 0.530516 |
| bowl3 | 0.955556 | 0.855255 |
| bowl3\|bulge | 1 | 0.955086 |
| bowl3\|concavity | 0.942857 | 0.832484 |
| bowl3\|scratch | 0.9 | 0.639366 |
| bowl4 | 1 | 0.734168 |
| bowl4\|bulge | 1 | 0.805815 |
| bowl4\|concavity | 1 | 0.812012 |
| bowl4\|scratch | 1 | 0.422547 |
| bowl5 | 0.884211 | 0.645095 |
| bowl5\|broken | 0.866667 | 0.566797 |
| bowl5\|bulge | 0.914286 | 0.665874 |
| bowl5\|concavity | 0.895238 | 0.660983 |
| bowl5\|hole | 0.833333 | 0.592889 |
| bowl5\|scratch | 0.733333 | 0.245375 |
| bucket0 | 0.949206 | 0.759023 |
| bucket0\|broken | 1 | 0.827687 |
| bucket0\|bulge | 1 | 0.836904 |
| bucket0\|concavity | 0.847619 | 0.80944 |
| bucket0\|crak | 1 | 0.360253 |
| bucket0\|hole | 1 | 0.795544 |
| bucket0\|scratch | 1 | 0.620731 |
| bucket1 | 0.873016 | 0.87787 |
| bucket1\|broken | 0.666667 | 0.777851 |
| bucket1\|bulge | 0.87619 | 0.854464 |
| bucket1\|concavity | 0.952381 | 0.938854 |
| bucket1\|crak | 1 | 0.425497 |
| bucket1\|hole | 0.833333 | 0.714966 |
| bucket1\|scratch | 0.533333 | 0.644734 |
| cap0 | 0.925926 | 0.90987 |
| cap0\|broken | 0.866667 | 0.850834 |
| cap0\|bulge | 0.87619 | 0.890177 |
| cap0\|concavity | 1 | 0.95993 |
| cap0\|hole | 0.9 | 0.915897 |
| cap3 | 0.97193 | 0.965336 |
| cap3\|bending | 1 | 0.99497 |
| cap3\|broken | 0.933333 | 0.922619 |
| cap3\|bulge | 0.942857 | 0.961105 |
| cap3\|concavity | 1 | 0.971099 |
| cap3\|hole | 1 | 0.98001 |
| cap4 | 0.968421 | 0.93938 |
| cap4\|bending | 1 | 0.990492 |
| cap4\|broken | 1 | 0.966726 |
| cap4\|bulge | 0.961905 | 0.916807 |
| cap4\|concavity | 0.961905 | 0.952002 |
| cap4\|hole | 0.966667 | 0.967294 |
| cap5 | 0.954386 | 0.933446 |
| cap5\|bending | 1 | 0.967775 |
| cap5\|broken | 0.8 | 0.848967 |
| cap5\|bulge | 0.980952 | 0.938863 |
| cap5\|concavity | 0.990476 | 0.941298 |
| cap5\|hole | 0.866667 | 0.723962 |
| cup0 | 0.985714 | 0.861293 |
| cup0\|bulge | 0.980952 | 0.838595 |
| cup0\|concavity | 0.990476 | 0.914248 |
| cup1 | 1 | 0.735271 |
| cup1\|bulge | 1 | 0.71608 |
| cup1\|concavity | 1 | 0.758117 |
| eraser0 | 0.880952 | 0.87012 |
| eraser0\|bulge | 0.771429 | 0.847092 |
| eraser0\|concavity | 0.990476 | 0.891903 |
| headset0 | 0.804444 | 0.731442 |
| headset0\|bending | 0.933333 | 0.909638 |
| headset0\|bulge | 0.8 | 0.640014 |
| headset0\|concavity | 0.790476 | 0.83813 |
| headset1 | 0.957143 | 0.731932 |
| headset1\|bulge | 0.990476 | 0.737233 |
| headset1\|concavity | 0.92381 | 0.753173 |
| helmet0 | 0.913043 | 0.79685 |
| helmet0\|bending | 1 | 0.720229 |
| helmet0\|broken | 1 | 0.679532 |
| helmet0\|bulge | 0.942857 | 0.897179 |
| helmet0\|concavity | 0.828571 | 0.8183 |
| helmet0\|crak | 0.866667 | 0.105913 |
| helmet0\|hole | 0.933333 | 0.329984 |
| helmet0\|scratch | 1 | 0.657819 |
| helmet1 | 1 | 0.615464 |
| helmet1\|bulge | 1 | 0.621174 |
| helmet1\|concavity | 1 | 0.608757 |
| helmet2 | 0.747826 | 0.878402 |
| helmet2\|bending | 0.866667 | 0.941038 |
| helmet2\|broken | 0.733333 | 0.993004 |
| helmet2\|bulge | 0.733333 | 0.947578 |
| helmet2\|concavity | 0.857143 | 0.93423 |
| helmet2\|crak | 0.766667 | 0.682557 |
| helmet2\|hole | 0.566667 | 0.580093 |
| helmet2\|scratch | 0.533333 | 0.651519 |
| helmet3 | 1 | 0.667767 |
| helmet3\|broken | 1 | 0.0643443 |
| helmet3\|bulge | 1 | 0.621837 |
| helmet3\|concavity | 1 | 0.750445 |
| helmet3\|crak | 1 | 0.0914993 |
| helmet3\|hole | 1 | 0.256677 |
| helmet3\|scratch | 1 | 0.818785 |
| jar0 | 0.97619 | 0.909678 |
| jar0\|bulge | 0.952381 | 0.886122 |
| jar0\|concavity | 1 | 0.947556 |
| microphone0 | 0.985714 | 0.898087 |
| microphone0\|bulge | 1 | 0.912981 |
| microphone0\|concavity | 0.971429 | 0.881227 |
| shelf0 | 0.782609 | 0.712568 |
| shelf0\|bending | 0.8 | 0.663835 |
| shelf0\|broken | 0.833333 | 0.390459 |
| shelf0\|bulge | 0.828571 | 0.814257 |
| shelf0\|concavity | 0.790476 | 0.658587 |
| shelf0\|crak | 0.9 | 0.25119 |
| shelf0\|hole | 0.566667 | 0.225338 |
| shelf0\|scratch | 0.633333 | 0.550755 |
| tap0 | 0.957576 | 0.622709 |
| tap0\|broken | 1 | 0.523318 |
| tap0\|bulge | 0.942857 | 0.53855 |
| tap0\|concavity | 0.961905 | 0.750784 |
| tap0\|crak | 1 | 0.183025 |
| tap0\|hole | 0.933333 | 0.189428 |
| tap0\|scratch | 0.933333 | 0.565097 |
| tap1 | 0.859259 | 0.611318 |
| tap1\|broken | 0.866667 | 0.570039 |
| tap1\|bulge | 0.838095 | 0.559186 |
| tap1\|concavity | 0.866667 | 0.687257 |
| tap1\|hole | 0.9 | 0.509744 |
| vase0 | 0.929167 | 0.91241 |
| vase0\|bulge | 0.838095 | 0.885184 |
| vase0\|concavity | 1 | 0.982515 |
| vase0\|scratch | 1 | 0.930962 |
| vase1 | 0.9 | 0.732615 |
| vase1\|bulge | 0.92381 | 0.730049 |
| vase1\|concavity | 0.87619 | 0.733679 |
| vase2 | 0.866667 | 0.849978 |
| vase2\|bulge | 0.895238 | 0.858266 |
| vase2\|concavity | 0.838095 | 0.852019 |
| vase3 | 0.757576 | 0.862079 |
| vase3\|broken | 0.6 | 0.836963 |
| vase3\|bulge | 0.72381 | 0.871578 |
| vase3\|concavity | 0.780952 | 0.891512 |
| vase3\|crak | 0.866667 | 0.779981 |
| vase3\|hole | 1 | 0.793726 |
| vase3\|scratch | 0.6 | 0.772522 |
| vase4 | 0.875758 | 0.883506 |
| vase4\|broken | 0.9 | 0.967484 |
| vase4\|bulge | 0.92381 | 0.926809 |
| vase4\|concavity | 0.780952 | 0.866639 |
| vase4\|crak | 1 | 0.75813 |
| vase4\|hole | 0.766667 | 0.392309 |
| vase4\|scratch | 1 | 0.890568 |
| vase5 | 1 | 0.695899 |
| vase5\|bulge | 1 | 0.711384 |
| vase5\|concavity | 1 | 0.685223 |
| vase7 | 1 | 0.745133 |
| vase7\|bulge | 1 | 0.736848 |
| vase7\|concavity | 1 | 0.756386 |
| vase8 | 0.848485 | 0.90603 |
| vase8\|broken | 0.6 | 0.979162 |
| vase8\|bulge | 0.885714 | 0.909134 |
| vase8\|concavity | 0.990476 | 0.958211 |
| vase8\|crak | 0.833333 | 0.759614 |
| vase8\|hole | 0.766667 | 0.559733 |
| vase8\|scratch | 0.566667 | 0.69685 |
| vase9 | 0.878788 | 0.821715 |
| vase9\|broken | 0.866667 | 0.986422 |
| vase9\|bulge | 0.866667 | 0.825208 |
| vase9\|concavity | 0.895238 | 0.840287 |
| vase9\|crak | 0.933333 | 0.338621 |
| vase9\|hole | 0.9 | 0.646975 |
| vase9\|scratch | 0.8 | 0.877198 |
| mean\|bending | 0.942857 | 0.883997 |
| mean\|broken | 0.85 | 0.725427 |
| mean\|bulge | 0.917143 | 0.82074 |
| mean\|concavity | 0.932619 | 0.845188 |
| mean\|crak | 0.915385 | 0.444732 |
| mean\|hole | 0.866667 | 0.609548 |
| mean\|scratch | 0.84152 | 0.660566 |
| mean | 0.920527 | 0.813001 |

### 2.5 MulSen-AD

Download the dataset from [here](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main) and process following [this guide](https://github.com/hzzzzzhappy/Processing-tools-for-the-MulSen_AD-dataset.git).

Set dataset paths in `./experiments/MulSen-AD/config.yaml`.

Training/Evaluation:
```bash
cd ./experiments/MulSen-AD/
sh train.sh 1 O # or sh eval.sh 1 0
```
Note: Multi-GPU training is not supported for evaluation, set `saver.load_path` in config.yaml.
