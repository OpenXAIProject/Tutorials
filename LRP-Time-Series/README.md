
LRP-Time-Series
==

Python implementation of the LRP method that is a novel methodology for interpreting generic multilayer neural networks by decomposing the network classification decision into contributions of its input elements.

## Reference Code 
Based on code by [Eric (Beomsu) Kim](https://github.com/1202kbs/Understanding-NN), [Chintan Zaveri](https://github.com/zaverichintan/HAR_prediction)

## Reference Paper 
**"Explaining nonlinear classification decisions with deep taylor decomposition"**. Gregoire Montavon, Sebastian Bach, Alexander Binder, Wojciech Samek, and Klaus-Robert Muller (https://arxiv.org/abs/1512.02479)

## Example Setup 
This is a deep learning method to classify time- series dataset. Our goal is to test how the LRP (more specifically deep Taylor Decomposition) can perform to depict the important time epochs and features from raw time series data. 
<p align="center"> 
<img src="https://github.com/OpenXAIProject/LRP-Time-Series/blob/master/result.jpg"  width="600">
</p>

## Dataset 
We will use the classic Human Activity Recognition (HAR) dataset from the UCI repository. The dataset contains the raw time-series data on human activity.
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

## Details of Dataset and Models 
+ We use deep neural network with four 1D convolution layers and 1 fully connected layer. 
+ In the code, cast the data set in a numpy array with shape (batch-size, sequence-len, n-channels) 
+ Batch-size: the # of examples training together 
+ Sequence-len: the length of sequence in time (128 steps here) 
+ N-channels: the # of channels in the layer (# of channels in input is the # of measurements) ties: 
+ There are 6 classes of activities: walking, walking upstairs, walking downstairs, sitting standing, laying
<p align="center"> 
<img src="https://github.com/OpenXAIProject/tutorials/blob/master/LRP-Time-Series/model.jpg" width="600">
</p>

## Installation
<img src="https://github.com/OpenXAIProject/tutorials/blob/master/LRP-Time-Series/howtorun.gif"  width="800">

**1. Fork & Clone** : Fork this project to your repository and clone to your work directory.
 
 ``` $ git clone https://github.com/OpenXAIProject/LRP-Time-Series.git ```
 
**2. Download Dataset** : Go to the `UCI` repository site and download the "UCI HAR Dataset" 
 
**3. Change Directory** : Move the "UCI HAR Dataset" to your work directory. It must be in the same folder as `LRP_tutorial.ipynb`.

**4. Run** : Run `LRP_tutorial.ipynb` or `LRP_tutorial.py`

## Requirements 
+ tensorflow (1.9.0)
+ numpy (1.15.0)
+ matplotlib (2.2.2)
+ scikit-learn (0.19.1)

## License
[Apache License 2.0](https://github.com/OpenXAIProject/tutorials/blob/master/LICENSE "Apache")

## Contacts
If you have any question, please contact Xie Qin (xieqin856@unist.ac.kr) and/or Sohee Cho (shcho@unist.ac.kr).

<br /> 
<br />

# XAI Project 

**This work was supported by Institute for Information & Communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence (의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics  

+ Web Site : <http://openXai.org>

