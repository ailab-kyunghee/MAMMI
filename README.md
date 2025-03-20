# [MICCAI2024] Mask-Free Neuron Concept Annotation for Interpreting Neural Networks in Medical Domain
[![🦢 - Paper](https://img.shields.io/badge/🦢-Paper-red)](https://arxiv.org/abs/2407.11375)  
This repo is the official source code for 'Mask-Free Neuron Concept Annotation for Interpreting Neural Networks in Medical Domain' International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)

## Introduction

## Preparation
1. Create virtual environment by conda.
```
conda create -n MAMMI python=3.10
conda activate MAMMI
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
* Note
    - MedCLIP: pip install git+https://github.com/RyanWangZf/MedCLIP.git
    - CLIP: pip install git+https://github.com/openai/CLIP.git

2. Prepare resources to run code.

    **Data**
    - Probing set:
    [NIH14](https://www.kaggle.com/datasets/nih-chest-xrays/data?select=test_list.txt), [ChestX-det](https://github.com/Deepwise-AILab/ChestX-Det-Dataset?tab=readme-ov-file) (for visualization)
    - Concept set: MIMIC-CXR Report (following [R2Gen](https://github.com/cuhksz-nlp/R2Gen?tab=readme-ov-file));
    We provide preprocessed concept set by MIMIC CXR Report test data. ('./dataset/report/nouns.txt)  
    Also, we povide processing code for concept set consturction. (`prepare_mimic_nouns.py`)

    **Pre-trained model**  
    Model[(Link)](https://github.com/lambert-x/medical_mae/tree/main): DenseNet121(Moco v2), ResNet50(Moco v2)  
    Put in `pretrained/target_model/{TARGET_MODEL.pth}`

## 1. Prepare Concept set (MIMIC Nouns)
We already provide concept set file. `dataset/report/nouns.txt`.  
If you want to create concept set, run 'prepare_mimic_nouns.py' 
* \# of MIMIC Nouns = 1361  

## 2. Example Selection 
run 'example_selection.py'

## 3. Concept matching
run 'concept_matching.py'

## Visualization
run 'target_model_perform.py' for multi-label classification.  
run 'bbox_img.py'  
run 'visualization.py'  

## Acknowledgement
This work was supported by the IITP grant funded by the Korea government (MSIT) 
(No.RS2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)).
