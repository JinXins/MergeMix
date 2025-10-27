<div align="center">
<h2><a href="https://arxiv.org/abs/2312.11954">MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding</a></h2>

[Xin Jin](https://scholar.google.com/citations?user=v3OwxWIAAAAJ&hl=zh-CN)<sup>1,\*</sup>, [Siyuan Li](https://scholar.google.com/citations?user=SKTQTXwAAAAJ&hl=zh-CN)<sup>1,2,\*</sup>, [Siyong Jian](https://scholar.google.com/citations?user=BodnjL0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Kai Yu](https://openreview.net/profile?id=~Kai_Yu12)<sup>1</sup>, [Huan Wang](https://scholar.google.com/citations?user=0-On0y4AAAAJ&hl=zh-CN)<sup>1,†</sup>,

<sup>1</sup>[Westlake University](https://www.westlake.edu.cn/)

<sup>2</sup>[Zhejiang University, College of Computer Science and Technology](http://www.cs.zju.edu.cn/)

<sup>*</sup> Equal Contribution <sup>†</sup> Corresponding Author
</div>

<p align="center">
<a href="https://arxiv.org/abs/2312.11954" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2312.11954-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/JinXins/MergeMix/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<a herf="" alt="Github stars">
    <img src="https://img.shields.io/github/stars/JinXins/MergeMix?color=blue" /></a>
</p>


## 📆 News  
**[2025/10/28]** We release the MergeMix code on LLaVA codebase.

**[2025/10/14]** We've updated our method on [**openmixup**](https://github.com/Westlake-AI/openmixup).


## 🛠 Installation

### 1️⃣ For Image Classification

You could use the openmixup environment and codebase if you want to use MergeMix on image classification task.

OpenMixup is an open source mixup toolbox and a benchmark for visual representation learning, and we highly recommend using it in your research if you want to use some mxiup augmentation techniques in image classification.

---
OpenMixup is compatible with **Python 3.6/3.7/3.8/3.9** and **PyTorch >= 1.6**. Here are quick installations for installation in the development mode:

```shell
conda create -n openmixup python=3.8 pytorch=1.12 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate openmixup
pip install openmim
mim install mmcv-full
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py develop
```

<details>
  <summary>Installation with PyTorch 2.x requiring different processes.</summary>

  ```bash
  conda create -n openmixup python=3.9
  conda activate openmixup
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl
  git clone https://github.com/Westlake-AI/openmixup.git
  cd openmixup
  pip install -r requirements/runtime.txt
  python setup.py develop
  ```
</details>

### Getting Started

OpenMixup supports Linux and macOS. It enables easy implementation and extensions of mixup data augmentation methods in existing supervised, self-, and semi-supervised visual recognition models. Please see [get_started.md](https://github.com/Westlake-AI/openmixup/blob/main/docs/en/get_started.md) for the basic usage of OpenMixup.

### Training and Evaluation Scripts

Here, we provide scripts for starting a quick end-to-end training with multiple `GPUs` and the specified `CONFIG_FILE`. 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
For example, you can run the script below to train a ResNet-50 classifier on ImageNet with 4 GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/classification/imagenet/resnet/resnet50_4xb64_cos_ep100.py 4
```
After training, you can test the trained models with the corresponding evaluation script:
```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${PATH_TO_MODEL} [optional arguments]
```

### 2️⃣ For LLaVA
We use the same environment as [**LLaVA**](https://github.com/haotian-liu/LLaVA).

### Getting Started

1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

### 🛫 Training
We use the LLaVA for our codebase:
1. Install the environment like LLaVA-v1.5
2. Find the bash `scripts/v1_5/finetune.sh`
3. Add some args in bash file as 
    - --use_augment True
    - --augment_type "mixup"
    - --use_ranking True
    - --use_tome True
    - --tome_ratio 0
    - --tome_merge_num 288
4. Run the bash file `scripts/v1_5/finetune.sh`

### 🛬 Inference
You need to find the `config.json` of the checkpoints, and add the parameters:
1. "use_tome": true
2. "tome_ratio": 0,
3. "tome_merge_num": 432


## ❤ Acknowledgement
- This work is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), and [OpenMixup](https://github.com/Westlake-AI/openmixup). We thank them for their excellent open-source contributions.

---
## 🤗 Citation
**If you feel that our work has contributed to your research, please consider cite it, and please don`t forget to cite OpenMixup if you use this project.**  
```markdown
@article{li2022openmixup,
  title = {OpenMixup: A Comprehensive Mixup Benchmark for Visual Classification},
  author = {Siyuan Li and Zedong Wang and Zicheng Liu and Di Wu and Cheng Tan and Stan Z. Li},
  journal = {ArXiv},
  year = {2022},
  volume = {abs/2209.04851}
}
```