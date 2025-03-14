# [MICCAI2024] Mask-Free Neuron Concept Annotation for Interpreting Neural Networks in Medical Domain

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
    - Concept set
    [MIMIC-CXR Report](https://github.com/cuhksz-nlp/R2Gen?tab=readme-ov-file); We provide preprocessed MIMIC CXR Report test data in this repo. ('./dataset/report')

    **Pre-trained model**  
    Model[(Link)](https://github.com/lambert-x/medical_mae/tree/main): DenseNet121(Moco v2), ResNet50(Moco v2)  
    Put in `pretrained/target_model/{TARGET_MODEL.pth}`

## 1. Prepare Concept set (MIMIC Nouns)
run 'prepare_mimic_nouns.py'
* \# of MIMIC Nouns = 1361

## 2. Example Selection 
run 'example_selection.py'

## 3. Concept matching
run 'concept_matching.py'

## Visualization
TBD

## Acknowledgement
This work was supported by the IITP grant funded by the Korea government (MSIT) 
(No.RS2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)).
