# Background research

http://femvestor.blogspot.com/2017/10/part-1-pitfalls-in-building-failures.html   
http://femvestor.blogspot.com/2017/10/part-2-rnn-model-to-predict-device.html   

## Hurst
https://github.com/dsdaveh/device-failure-analysis   

#### EDA and Feature Engineering
*  analyze this data at the device grain
* Over 400 function devices are taken from the log in the first 5-6 days. The first failure doesn't occur until day 5.
* double check that devices can only fail once: TRUE
* there are 5 zombie drives: most drives are removed on the day failure is indicated, a few do not (see the plot below).
* The attributes are all integers, but many have the appearance of being error codes, with a large percentage of 0 values and large gaps between frequent occurances. This is common with devices that may have error codes that are bit encoded (eg. 2,4,8,16...1024, etc.).
* develop categorical variables with more predictive power than treating the attributes as integer
* 6 attributes have very high percentage of zero values. These could be error codes or some other indicator
* many of the attributes are heavily skewed. These should be transformed for some analyses although this wasn't necessary for my models.

#### Time Series
* By transforming the dataset at the device grain, we have have a 9.08% failure rate

#### modeling

## kashyap
https://github.com/kashyap16/Classification-predict_failure   

#### EDA and Feature Engineering
* heatmap: attrute 9 and 3 seem to have a good co-relation though not high enough
* resampling data for imbalanced classes

#### Modeling
* train-test split
* accuracy, recall, conf matrix, roc auc
* decision tree classifier, random forest, naive bayes, svm
* cool model comparison

## gdhruv80
https://github.com/gdhruv80/Hazard-Modelling-Time-to-device-failure    

#### EDA and Feature Engineering
* Basically its a time to event analysis: model the survival time.
* What does the lifecycle of a device look like - Is the hazard rate really high in the begining and end and kind on low during the most of the life in the middle just like say with a bulb/hardisk
* Are there gaps between dates when values are recorded for a device
* there are roughly 15% of the devices where the timeline is not continous
* aggregating the data at a weekly level to remove too much noise
* If the bulb is tested on any day of the week it is assumed to be good for the week
* Populating correct nth Day from Begining - (Accounting for time gaps in recordings for a device)
* we will have to create 2 new variables T0 and T1 which reflect the begin time and end time for each row in the dataset for the device
* Note that the Time interval for each row is (T0-T1]
* Attribute 3 and 9 have significant +ve correlation(good to know)
* for 6 of the 9 variables more than 75% of the values are 0
* Capping outliers to the 1.5 IQR value (this is better than imputing with the median)
* including interaction terms of attribute 2 and 4 with time (T0). This should accomodate for the changing hazard ratios.

#### Modeling
* Cox - Regression which does not force any assumptions on the shape of underlying data
* train, test, and validation sets
* Building a final model based on the 3 attributes selected above
* The model has a R(square) of 0.002 and is significant based on all 3 tests (Log-Liklihood,Wald and Logrank)
* "the probability of survivial at the 300 day mark goes down from 59% to 40%."
* Checking propotional hazard assumption (if HR is constant over time or not) --> requires time interactions
* ROC AUC score = .72, confusion matrix, optimal cutoff point, lift calculation

## AVJ
https://github.com/AVJdataminer/Sensor   

#### EDA and Feature Engineering
* Failures and Percent Failures by Month
* group by device
* failure is most highly correlated with attributes; 2,4,7,& 8
* outliers: remove data that is outside 1.5 times the interquartile range.
* replace the NA's with medians calculated excluding those outliers and replace NA's with the medians.
* review the level of variance within the variable: Remove Variables with Near Zero Variance
* Rescale and Center the Data for Modeling
* Find Variables That are 100% Unique values: These need to be removed, usually a count or id variable.
* Could create PCA or KNN to get more features.

#### Modeling
* train-test split
* logistic regression, random forest, MxNET neural net
* AUC, conf matrix, accuracy
* Neural Network Model Performance on holdout Test set
* Random Forest Model has best overall performance.

## Kyamz
https://github.com/kyamz/ADS_Class   

#### EDA and Feature Engineering
* Does all her work on AWS
* Groups by device (1168 rows)
* About 9% of devices have a recorded failure. We will likely need to group by devices to help with the class imbalance.
* Looking at within-device variance for each feature: Attributes 2, 3, 4, 7 and 9 all have a variance of zero for over 75% of devices.
* Rare variance: 164 of the 934 total devices have 'rare variance'. 63 of the 85 failing devices have this feature
* only 101 of the 849 non-failing devices have variance in attributes 2, 3, 4, 7 and 9.
* 3 devices with start dates other than Jan 1, 2015.
* Capture interactions.
* Clean up a few outliers.
* Outcome: Device fails permanently.

#### Time-series data
* Needs data _every day_. Are there periodic gaps? Yes.
* Extensive gaps don't correlate with failure, but they do with rare variance.
* Ideally we would want every device to either make it to the end of dataset's time period or end in failure.
* What should we do with the devices that don't end in failure but whose data record ends abruptly? Most of data.
* group all the data by device ID and make our modeling strategy to build a binary classifier targeting if the device will have a failure record or not.
* risky if we define our target model to be something like 'failure will happen within next week, month, etc).

#### Device ID groups
* ID string is a feature: S, W, and Z.
* W-type devices are more likely to fail than the other two groups.
* A pairplot also demonstrates that there are major differences in the sensor data across groups: ie intersection of attributes 1&3.

#### Modeling
* splits into training, validation, and test sets.
* XGBoost.
* hyperparameter tuning.
* ROC AUC .94, f1 score .68
* feature importances

### Critique

## Mohammad
https://gist.github.com/mohammadbutt/3659d0564ce41220a38e9cd2be282593   

#### EDA and Feature Engineering
* heatmap correlations
* creates new dataframe: one row for each device (reduces dimensions from 124K to 1168)
* joins groupby dataset to regular dataset
* converts data from string to dt. creates month.
* Nice scatter plots of working/failing devices
* convert categorical features to dummies, then create interaction terms!!

#### Modeling
* Uses SMOTE for imbalanced classes
* train-test split, crossvalidation
* RF, GBC, logistic, KNN classifiers; voting ensemble
* metric: accuracy, precision, recall. F1 score 98%.
* displays feature importance

#### Critique
* using groupby MAX loses lots of information from the features.
* Doesn't account for time-series data.




## Nolan
https://granolanbar.github.io/projects/   

#### EDA and Feature Engineering
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

#### EDA and Feature Engineering
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

#### EDA and Feature Engineering
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


## My reflections

#### What is the research question?
* "For any given device, will it _ever_ fail at any point during its life?" (classification)
* "For any given device, how many days from launch until failure?" (regression)
* "For each device, on any given day, what is the probability that the device will fail _tomorrow_?" (many, many classifications: 1168x304) -- most interesting, but also hardest.

#### Feature engineering
* Time since launch, in days.
* First 3 digits of ID code
* Rolling average of previous status codes
* If there are empty "interval" days, do I need to create them and impute averages? (this is essentially missing data).
* convert categorical features to dummies, then create interaction terms!!
* Rare variance: 164 of the 934 total devices have 'rare variance'. 63 of the 85 failing devices have this feature

#### Reshape the data
* Do I need 1168 rows, and then multiple blocks of columns (one block for each day?)
* Do I need multiple datasets - one per day, each with 68 rows, 9 feature columns, plus a target (tomorrow's failure)?
