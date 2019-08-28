# Predict Device Failure

[Model Results on AWS](http://python-eda-example.us-east-1.elasticbeanstalk.com/)

Challenge: A large fleet of devices requires maintenance to prevent device failure. This repository presents a predictive analysis to help with device maintenance by predicting failure given a series of device attributes, measured daily over the course of 11 months in 2015. The final analysis employs a neural network model with a F1 Score of .96 and a ROC-AUC score of .86.

Dataset: 124,164 daily readings from 1163 devices across 9 attributes related to device failure.

## Steps in Analysis:  

* [1. Data cleaning](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/01_data_cleaning.ipynb) 
* [2. EDA by device ID](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/02_devices.ipynb) 
* [3. Trimming the dataset](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/03_trimming.ipynb)  
* [4. EDA by other variables](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/04_EDA.ipynb) 
* [5. Feature engineering](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/05_feature_engineering.ipynb)  
* [6. Logistic Regression Results](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/06_modeling.ipynb) 
* [7. Neural Network Results](https://nbviewer.jupyter.org/github/austinlasseter/predict_device_failure/blob/master/analysis/07_tensorflow.ipynb)

<div align="center">
    <img src="https://github.com/austinlasseter/predict_device_failure/blob/master/images/net_metric.png" width="1000px"</img> 
</div>

<div align="center">
    <img src="https://github.com/austinlasseter/predict_device_failure/blob/master/images/net_rocauc.png" width="1000px"</img> 
</div>

[Problem Description](https://drive.google.com/open?id=0B_cz06nPiN5CVk1qci1EQUhyM3JON0lROGVZWmJoelR2aHFV)

