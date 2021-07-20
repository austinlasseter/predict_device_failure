# Background research

## Wahl
https://gallery.azure.ai/Notebook/Predictive-Maintenance-Modelling-Guide-Python-Notebook-1#x_Data-Sources
* Telemetry data almost always comes with time-stamps which makes it suitable for calculating lagging features. A common method is to pick a window size for the lag features to be created and compute rolling aggregate measures such as mean, standard deviation, minimum, maximum, etc. to represent the short term history of the telemetry over the lag window. In the following, rolling mean and standard deviation of the telemetry data over the last 3 hour lag window is calculated for every 3 hours.
* add blank entries for all other hourly timepoints (since no errors occurred at those times)
* Predictive models have no advance knowledge of future chronological trends: in practice, such trends are likely to exist and to adversely impact the model's performance. To obtain an accurate assessment of a predictive model's performance, we recommend training on older records and validating/testing using newer records.
* Imbalanced classes: To help with this problem, sampling techniques such as oversampling of the minority examples are usually used.

## zaradski
https://towardsdatascience.com/water-pumps-maintenance-prediction-data-science-illustrated-20c7100017c5

* Trimming high-cardinality categorical variables: group all rare (i.e. being observed less than 40 times in a dataset of >50,000 records ) categorical values into a single “rare” value.
* Encoding categorical variables: we perform this conversion using the “pandas.get_dummies” function 
* Identifying highly-correlated features: to avoid “co-linearity” issues during the model estimation procedure and to reduce the dimension of the model optimization problem we should drop some of the near-equivalent columns.
* Identifying key predictive features: Identifying the variables the have significant predictive power on the target variable is similar to identifying “correlated” variables. 
* Interaction terms: If your dataset predictive variables have a lot of useful interactions (see the notebook) these models are likely to outperform the logistic regression accuracy : Random Forest (covered in the notebook) and XGBoost.

## femvestor
http://femvestor.blogspot.com/2017/10/part-1-pitfalls-in-building-failures.html   
http://femvestor.blogspot.com/2017/10/part-2-rnn-model-to-predict-device.html   

The goal of this task is to predict failing devices

#### EDA
* start at jan 3rd: there are 7 devices prior to that, which get cut off (no failure).
* At the end of the period, only 27 devices are left.
* Except for attributes 1,5, and 6, the other attributes have mostly zeroes. 
* Attribute 1 is just a signal emitted by the devices.  When a device fails the signal will stop at its current value.  Therefore, I decided to remove it. No sig difference between signal and failure.
* Attributes differ in their magnitudes.  Hence, scaling or centering may be required
* Some devices are removed and then put back at different time period.  This is a problem if one is using RNN where time is taken into consideration. (is this gap in monitoring?)
* Except for 4 devices, when a device fails it is removed
* most devices are removed or fail before 20 days.
* failed devices have a high failure up to 40 days, and then around 130 days.

#### Devices removed and put back
* Many working devices have been removed before 100 days of functioning.
* how many devices have been removed and put back?  
* given that I found 521 devices that were removed and then added at a later time means that these devices have to be removed when building the RNN.
* left with 1013 devices but 91 failing devices.
* Devices removed without any failure: remove all (this is almost the entire dataset). left with about 125 devices total.

#### Feature Engineering
* there's a sig correlation between failure and attributes (sig diff in means, etc)
* I can ignore the failure day observations, and just use the 4 previous days (t-4, t-3, t-2, and t-1) to see if I can predict failure at time t.
* a sequence of 4 days (t-4 to t-1) that map to an outcome at t.
* in all the last 10 days, before November 2nd, 2015, the model was starting to predict possible anomalies in these devices.  This may explain why the operators removed them on that day.

#### modeling
* use RNN, more precisely an LSTM (Long Short Term Memory),  as the Deep Learning model of choice
* the training set contains devices that failed on or before July 12th, 2015, and the testing set contains devices that failed after July 12th, 2015
* recall and precision, as we know it, hid the fact the model performs well.
* predicts tomorrow's failure on every day of each device.
* Out of all the 45 devices, the model accuracy was 91.30%.  However, for failed devices, it had a recall of  78.94%.  On the other hand, for working devices, it had a precision of 84.38%.

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
* Attribute1 was also given the max Z score for the last 4 days. 
* I also recorded the total number of records for a device (a proxy for service life)
* maximum gap between log entries.
* retirement data and the number of total failures for the 5 days prior are also calculated, but they were not used in the model

#### Additional feature engineering
* many of the attributes are heavily skewed. These should be transformed for some analyses although this wasn't necessary for my models.
* Each device gets multiple features for each variable.
* I included the mean and standard deviation for the entire device life 
* as well as the last value and difference between first and last values (drift)

#### Time Series
* By transforming the dataset at the device grain, we have have a 9.08% failure rate
* looking for changes to baseline operation metrics should be a good indicator of failure (i.e., variance)

#### modeling
* random forest
* 10fold cross validation
* ROC AUC=.95, F1, accuracy etc. as function of various cutoffs
* This model catches about half the cases using a cutoff of 0.5
* Drift for attribute 4 was almost always the primary indicator of failure across all folds and models. 
* After that n_rec and and variance for attributes 6 and 7 were also important, but switched rank between the different folds/models.
* There's not enough data to train a DL model with any efficacy
* cost-benefit analysis of cut-off point.

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

#### Collapse on device ID?
* If there is no variance at all other than date, then there's no need to keep all this data.
* If there's only a little variance, then we can capture the std dev of each, and tag certain devices as "high variance."
* This is variance at the device level, not at the column level (ie, for each device, little/no change per var over life)
  - How would I calculate this? "High variance device."
