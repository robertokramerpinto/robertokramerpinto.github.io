# Model Validation

Model validations is any machine learning framework developed to ensure your model performs as expected when facing new data.
Model validation can be performed in several ways, but the general idea is to assess your model's performance on new data: data that 
wasn't used during the training phase. 

Model validation is not only about setting different datasets to test the model's performance but it's also about:
* Selecting the best model (algorithm)
* Selecting the best parameters
* Selecting the best subset of features
* Comparing results based on appropriate metrics

> Model Validation Objective
* Select best model possible --> best accuracy for new data

**Train x Test Data**

Models tend to have higher accuracy on train data (on observations they have seen before). When models perform 
differently on training and testing data, you should look to model validation to ensure you have the
best performing model. 

## Bias-Variance Tradeoff

Whenever we're assessing a model's performance it's indicate to understand its variance and bias. In general terms, 
a model's error can be decomposed into 3 parts: variance_error, bias_error, general_error. The decomposition of the 
loss (error) helps us understand the learning algorithms, as these concepts are related to over/under fitting.

In general terms:
* High Variance --> Indicates **Overfitting**
* High Bias --> Indicates **Underfitting**

In high-level terms, we can say that:
> Bias
* Ability of the model to learn from data in order to make good predictions
* In average, how far is your model from the observed values?
* A high bias-error model indicates a low predictive power --> underfitting
    * model is not able to do its function

![](/assets/ml/theory/6.png)

> Variance
* Ability of the model to keep performance when facing a new set of data
* Does you model's predictions vary too much when facing new data?
* How stable is your model when facing different test sets?
* A high variance-error model indicates overfitting
    * Your model's results vary a lot when it faces different datasets

![](/assets/ml/theory/5.png)

**What is the trade-off?**

The trade-off comes from the fact that generally it's not possible to improve both aspects together. If you want to reduce
your bias, you'll increase your variance error. If you want to reduce your variance error, you'll increase your bias. 

The goal is to find an optimal solution.

![](/assets/ml/theory/7.png)

#### Identifying Bias and Variance

We can use the mlxtend library.

For comparison, the bias-variance decomposition of a bagging classifier, which should intuitively have a lower variance 
compared than a single decision tree:

````python
from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split


X, y = iris_data()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y)



tree = DecisionTreeClassifier(random_state=123)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        tree, X_train, y_train, X_test, y_test, 
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
>>Average expected loss: 0.062
>>Average bias: 0.022
>>Average variance: 0.040

from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(random_state=123)
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=100,
                        random_state=123)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        bag, X_train, y_train, X_test, y_test, 
        loss='0-1_loss',
        random_seed=123)

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)
>>Average expected loss: 0.048
>>Average bias: 0.022
>>Average variance: 0.026
````

## Overfitting

Occurs when our model is not able to generalize well to new (unseen) datasets.

When a model is overfitting, it's memorizing patterns to a specific training dataset and generally presents very high
accuracy metrics (for this training set) but it fails to predict information for new data. We se a drop in accuracy 
when assessing the model's performance over new data.

It's usually very impacted by the model's parameters selection. When we work with tree models, 
a higher number of estimators or max_depth, for example, can lead to some level of overfitting. 

## Train, Validation & Test : Holdout samples

The basic validation approach is to build holdout samples. The most indicated framework when using this path is to 
create 3 sets of data: training, validation & test sets.

> Training data
* data sample that will be used as the training reference

> Validation data
* Data sample used to tune hyperparameters, feature selection, etc...
* Model with different characteristics are trained over the training set and evaluated over the validation set
* Validation set is often used to select the best model and parameters

> Test Set
* After selecting the best model using the training + validation sets, usually we train the selected model over the 
entire data (training + validation) and assess it's expected performance using the test set. 
* Notice that the model selection and evaluation should be done using the validation data
* The test set can't be used to select the best model
* It's only used to serve as a reference for the model's performance over new data

**Split Ratios**

It depends on your problem but generally we use 60-20-20.

![](/assets/ml/theory/2.png)

### Implementation

In python, we can implement this solution through the train_test_split function (sklearn.model_selection module). To
obtain the training, validation and test samples, we need to run this method 2x. 

````python
from sklearn.model_selection import train_test_split
SEED = 42

# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  = train_test_split(X=X, y=y, test_size=0.20, random_state=SEED)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val  = train_test_split(X=X_temp, y=y_temp,test_size=0.20, random_state=SEED)

````

## Cross-Validation
Cross validation is a validation framework created to over come problems from the traditional holdout strategy.

One of the major problems with the holdout sample framework is the random effect. Because we randomly select part of the
data to be the training, validation & test datasets, in a worst case scenario we might just be lucky and select a test set
which is similar to the train set and we see good results. However, if new data comes into our model and it's different
from the test set, our model's performance might be not as good as expected. 

In other words, the holdout sample approach might be misleading due to randomness effects. 

Another problem occurs if we have a small dataset. If we split data into train, validation and test sets we might end
with lower number of observations to both train and evaluate our models.

One approach to address these issues is to use Cross-Validation.

Cross-Validation uses the entire dataset to train our models by performing multiple training and validation splits
over the dataset.

![](/assets/ml/theory/8.png)

In Python we can use the ````sklearn.model_selection```` module to build our Cross validation framework.

### Types of Cross Validation

#### KFold
KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), 
of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.

````python
>>> import numpy as np
>>> from sklearn.model_selection import KFold

>>> X = ["a", "b", "c", "d"]
>>> kf = KFold(n_splits=2)
>>> for train, test in kf.split(X):
...     print("%s %s" % (train, test))
[2 3] [0 1]
[0 1] [2 3]
````

#### Repeated K-Fold
RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, 
producing different splits in each repetition

`````python
>>> import numpy as np
>>> from sklearn.model_selection import RepeatedKFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> random_state = 12883823
>>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
>>> for train, test in rkf.split(X):
...     print("%s %s" % (train, test))
...
[2 3] [0 1]
[0 1] [2 3]
[0 2] [1 3]
[1 3] [0 2]
`````

#### Leave One Out (LOO) & Leave P Out (LPO)
LeaveOneOut (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one, 
the test set being the sample left out. LeavePOut is very similar to LeaveOneOut as it creates all the possible 
training/test sets by removing  samples from the complete set. 

````python
>>> from sklearn.model_selection import LeaveOneOut

>>> X = [1, 2, 3, 4]
>>> loo = LeaveOneOut()
>>> for train, test in loo.split(X):
...     print("%s %s" % (train, test))
[1 2 3] [0]
[0 2 3] [1]
[0 1 3] [2]
[0 1 2] [3]
```` 
#### Stratified KFold
StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately the 
same percentage of samples of each target class as the complete set.
````python
from sklearn.model_selection import StratifiedKFold, KFold
>>> import numpy as np
>>> X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
>>> skf = StratifiedKFold(n_splits=3)
>>> for train, test in skf.split(X, y):
...     print('train -  {}   |   test -  {}'.format(
...         np.bincount(y[train]), np.bincount(y[test])))
train -  [30  3]   |   test -  [15  2]
train -  [30  3]   |   test -  [15  2]
train -  [30  4]   |   test -  [15  1]
>>> kf = KFold(n_splits=3)
>>> for train, test in kf.split(X, y):
...     print('train -  {}   |   test -  {}'.format(
...         np.bincount(y[train]), np.bincount(y[test])))
train -  [28  5]   |   test -  [17]
train -  [28  5]   |   test -  [17]
train -  [34]   |   test -  [11  5]
````

### General Python framework
````python
# Pure Python CV Framework
import pandas as pd
from sklearn import model_selection

def create_cv_dataset(data,target_name,n_folds=5,seed=42,stratified=True):
  df = data.copy()
  
  # Initializing a Fold Column
  df['kfold'] = -1
  # Randomize data
  df = df.sample(frac=1).reset_index(drop=True)
  
  y = df[target_name].values
  
  # Initiate CV object
  if stratified:
    kf = model_selection.StratifiedKFold(n_splits=n_folds)
  else:
    kf = model_selection.KFold(n_splits=n_folds)
  
  # Fill kfold column
  for f, (t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f
  
  return df

# Application Example
cv_python_data = create_cv_dataset(data = pandas_train, target_name = 'Response')

````

Getting scores direclty with cross_validation_score
`````python
# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=make_scorer(mean_squared_error))

# Print the mean error
print(cv.mean())
`````

After getting and validating data --> Indicated to persist
* Same data can be used with several ML approaches & frameworks --> unique measurement comparison

It's also indicated to use a general train & test initial split. Over the training data we perform our cross-validation
framework so we can define the best parameters, transformations and featuers. After selecting the best model (comparing
average results over the validation folds) we can then train the model over the entire train data and assess its 
performance on the test set. 
 

### PySpark framework
This is a Pyspark Stratified Cross-Validation framework implementation.

````python
# Pyspark Framework
def create_spark_cv_dataset(data,target_name,n_folds=5,seed=42,stratified=True):
  """ Binary Classification only : Target needs to be numeric (1 and 0)
  """
  split_ratio = 1.0 / n_folds
  
  positives = data.where(F.col(target_name) == 1)
  negatives = data.where(F.col(target_name) == 0)
  
  positive_splits = positives.randomSplit([split_ratio for i in range(n_folds)])
  negative_splits = negatives.randomSplit([split_ratio for i in range(n_folds)])
  
  for i in range(n_folds):
    sample_df = positive_splits[i].union(negative_splits[i]).withColumn('kfold',F.lit(f'{i}'))
    if i == 0:
      final_df = sample_df
    else:
      final_df = final_df.union(sample_df)

  return final_df

final_df = create_spark_cv_dataset(train,target_name = 'Response')
# Check no if is repeated in different folds
final_df.groupBy('id','kfold').agg(F.count(F.col('kfold')).alias('count')).where(F.col('count')>1).show()
>> Needs to be None
# Check target distribution among folds
final_df.groupBy('kfold').agg(F.avg(F.col('Response'))).show()
````

### Time Series Cross Validation

Time series data is characterised by the correlation between observations that are near in time (autocorrelation). 

However, classical cross-validation techniques such as KFold and ShuffleSplit assume the samples are independent 
and identically distributed, and would result in unreasonable correlation between training and testing instances 
(yielding poor estimates of generalisation error) on time series data. 

Therefore, it is very important to evaluate our model for time series data on the “future” observations least 
like those that are used to train the model.

In sklearn we can use the TimeSeriesSplit class to determine such framework.

TimeSeriesSplit is a variation of k-fold which returns first  folds as train set and the  
th fold as test set. Note that unlike standard cross-validation methods, successive training sets are supersets of 
those that come before them. Also, it adds all surplus data to the first training partition, which is always used to train the model.

This class can be used to cross-validate time series data samples that are observed at fixed time intervals.

```python
>>> from sklearn.model_selection import TimeSeriesSplit

>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> tscv = TimeSeriesSplit(n_splits=3)
>>> print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=3)
>>> for train, test in tscv.split(X):
...     print("%s %s" % (train, test))
[0 1 2] [3]
[0 1 2 3] [4]
[0 1 2 3 4] [5]
```
![](/assets/ml/theory/9.png)




