# Easy3AD
Hope you find this a useful baseline! The framework is simple and easy to extend for your own projects. This is the official implementation of the paper "Easy3AD: A Unified 3D Anomaly Detection Method Based on Group-level Feature Fusion Reconstruction for Diversity Constraint". Any questions, please email Hanzhe Liang via [2023362051@email.szu.edu.cn](2023362051@email.szu.edu.cn).

Hanzhe Liang, Can Gao, Linlin Shen, Jinbao Wang.

> Unified 3D anomaly detection using reconstruction paradigms has attracted widespread attention from the research community due to its superior performance. Existing 3D anomaly detection methods directly employ encoder-decoder architectures to process pre-trained features, which preserves the diversity of original features during reconstruction, leading to identity mapping problems and making it difficult to distinguish between normal and abnormal features effectively. Additionally, while multi-scale information provides richer feature representations, effectively utilizing this information while avoiding identity mapping issues remains a significant challenge. We propose a deliberately simplified model Easy3AD designed to constrain feature diversity, which consists of the following three components. First, Multi-Scale Feature Fusion (MFF) extracts common features across multiple scales from pre-trained representations. Next, Spatial Channel Feature Aggregation (SCFA) merges redundant features across channels to constrain feature diversity and address the identity mapping problem. Finally, the Local Pattern Reconstruction Module (LPRM) progressively refines features through coarse-to-fine reconstruction for effective anomaly detection.

Unlike our previous MC3D-AD/C3D-AD work, we're going for a simpler approach here. The idea is to reduce feature complexity by combining different feature channels at multiple scales, then use progressive reconstruction to spot anomalies. 

## 1. Lucky Inspiration
This work builds upon our previous work; for more inspiration, please refer to [C3D-AD](https://arxiv.org/pdf/2508.01311) and [MC3D-AD](https://arxiv.org/pdf/2505.01969). The experimental data in the paper are easy to reproduce, but you will typically need to spend several hours or more than ten hours on training. If you need our pre-trained checkpoints, please reach out to the author. In addition, this work has been used in our upcoming review and benchmark. You can also check ["our lucky sheep"](http://de.com/hzzzzzhappy/FindWolf) out for inspiration.

## 2. Quick Start

### 2.1 Requirements
```bash
conda create -n Easy3AD python=3.8
conda activate Easy3AD
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

### 2.5 MulSen-AD

Download the dataset from [here](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main) and process following [this guide](https://github.com/hzzzzzhappy/Processing-tools-for-the-MulSen_AD-dataset.git).

Set dataset paths in `./experiments/MulSen-AD/config.yaml`.

Training/Evaluation:
```bash
cd ./experiments/MulSen-AD/
sh train.sh 1 O # or sh eval.sh 1 0
```
Note: Multi-gpu training is not supported. For evaluation, set `saver.load_path` in config.yaml.

## 3. Licence
MIT License

Copyright (c) 2025 Hanzhe Liang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

