# Background research

http://femvestor.blogspot.com/2017/10/part-1-pitfalls-in-building-failures.html
http://femvestor.blogspot.com/2017/10/part-2-rnn-model-to-predict-device.html
https://github.com/dsdaveh/device-failure-analysis
https://github.com/kashyap16/Classification-predict_failure
https://github.com/gdhruv80/Hazard-Modelling-Time-to-device-failure
https://github.com/AVJdataminer/Sensor
https://github.com/kyamz/ADS_Class
https://gist.github.com/mohammadbutt/3659d0564ce41220a38e9cd2be282593   


## Nolan
https://granolanbar.github.io/projects/   

#### Descriptive Stats
* Heatmap correlation matrix  
* measurement are taken until the device fails, since the last measurement is the failure
* Creating a column to measure time called 'daysActive
* Season (similar to month)


#### Modeling
* train-test split
* oversampling to account for imbalanced classes (SMOTE)
* Accuracy, precision, recall, confusion matrix, AUC
* kfold crossval
* logistic regression, ridge, stochastic gradient descent classifier
* gradient boost, random forest gets AUC of 83%

##### Critique
* Better to use Lasso regression to identify meaningful features, prevent overfit
* Fails to account for time series data




## Nguyen
https://duymnguyen9.github.io/Telemetry-Device-Failure-Machine-Learning/    

#### Descriptive stats
* Month
* Use `groupby` to create features (mean, sum)
* Use PCA to reduce attributes 3 and 4 into a single feature
* Imbalanced classes


#### Modeling
* kmeans crossvalidation (uses entire dataset)
* F1, precision, recall
* train-test split
* tried XGBoost, Gradient Boost, AdaBoost. final model was XGB.
* Display feature importance
* F1 score: 84.8%

#### Critique
* Does not address time-series element
* No real accounting for imbalanced classes
* Fails to eliminate attribute 8



## Huiming
http://songhuiming.github.io/pages/2017/09/23/data-engineering-and-modeling-01-predict-defaults-with-imbalanced-data/

#### Descriptive Stats
* x1 and x6 are like numeric variable while the others are like categorical variable  
* x2, x3, x7, x8, x9 has lots of zeros  
* get the time duration from first recording time to positive target time   
* There are only 3 different values for the first 2 strings in id. I guess it is like geo info.  
* By id it shows w1 has higher default rate then z1. And z1 is higher than s1. 
* We can see there is high correlation between x7 and x8. Alao x9 and 3 has a correlation around 0.53.   

#### There are some methods to deal with the imbalanced data:
* oversampling: we repeat the low proportion data to make the proportion of target=1 and target=0 to be close in the oversampled data.
* downsampling: sample from high proportion(target=1 here) to make the data balanced
* adjust the low proportion data weight in the algorithm
* adjust the decision threshold of the output probability to classify
* adjust the loss function if we want to give more weights on the low proportion data

#### Modeling
* Random Forest Classifier, ExtraTreeClassifier, AdaboostClassifier, GradientBoostingClassifier, LinearSVC, Logistic Regression
* Kfold crossvalidation, XGBoost  
* Display feature importance  
* Metric: Confusion Matrix, high accuracy (not reported)

#### Critique
* Doesn't consider time-series element at all  
* Hardly any feature engineering
* Fails to address imbalanced classes
