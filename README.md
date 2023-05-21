# DWFormer: Dynamic Window Transformer for Speech Emotion Recognition

This work is accepted by ICASSP 2023.

The paper link is: 

https://ieeexplore.ieee.org/abstract/document/10094651

https://arxiv.org/abs/2303.01694
## Data and Pretrained Model Preparation
The feature we used is from WavLM-Large, which could be downloaded from https://github.com/microsoft/unilm/tree/master/wavlm.

Download the IEMOCAP dataset from https://sail.usc.edu/iemocap/

Download the Meld dataset from https://affective-meld.github.io

## Feature extraction:
Feature extraction file is in ./Feature_extractor/data_preprocess.py

To process data in batches, we pad the data into the same length. The length of the data in IEMOCAP is 324, while the length in Meld
 is 226.
 
> python data_preprocess.py

## Training & Evaluating:
The files are IEMOCAP/train.py & MELD/train.py

The name of the model file is model.py

> python train.py

## Citation:

S. Chen, X. Xing, W. Zhang, W. Chen and X. Xu, "DWFormer: Dynamic Window Transformer for Speech Emotion Recognition," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10094651.
