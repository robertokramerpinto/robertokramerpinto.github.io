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


## Overfitting


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

### Python framework
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

After getting and validating data --> Indicated to persist
* Same data can be used with several ML approaches & frameworks --> unique measurement comparison 

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





