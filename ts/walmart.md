# Case: Walmart Store Sales Forecasting

https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

## Data Description

> Objective
* predict the department-wide sales for each store

> Features
* Historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments
* Walmart runs several promotional markdown events throughout the year. 
* These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. 
* The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. 

Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.

> Files
* stores.csv
    * This file contains anonymized information about the 45 stores, indicating the type and size of store
* train.csv
    * This is the historical training data, which covers to 2010-02-05 to 2012-11-01. 
    * Within this file you will find the following fields:
* test.csv
    * Output file expecting predictions
* features.csv
    * Additional data related to the store, department and region

## ETL Initial Data

Because this data contains a considerate number of rows and columns, the ETL part will processed in Spark (Databricks Cluster).

````python
#!kaggle competitions download -c walmart-recruiting-store-sales-forecasting
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
import numpy as np

import datetime

# Raw data were manually created in Databricks environment
# Loading Tables
train = spark.sql("SELECT * FROM walmart_train")
test = spark.sql("SELECT * FROM walmart_test")
stores = spark.sql("SELECT * FROM walmart_stores")
features = spark.sql("SELECT * FROM walmart_features")

# Create Holiday DF
holidays_pandas = pd.DataFrame({"Date":[datetime.datetime(2010,2,12),datetime.datetime(2011,2,11),datetime.datetime(2012,2,10),datetime.datetime(2013,2,8),
                                        datetime.datetime(2010,9,10), datetime.datetime(2011,9,9),datetime.datetime(2012,9,7),datetime.datetime(2013,9,6),
                                        datetime.datetime(2010,11,26),datetime.datetime(2011,11,25),datetime.datetime(2012,11,23),datetime.datetime(2013,11,29),
                                        datetime.datetime(2010,12,31),datetime.datetime(2010,12,30),datetime.datetime(2012,12,28),datetime.datetime(2013,12,27)],
                               "fl_superbowl":[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                               "fl_tksgiving": [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                               "fl_xtmas": [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                               "fl_laborday":[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0]})

def preprocess_data(data, stores, features, holidays):
  
  # Casting TimeStamp to Date
  data = data.withColumn('Date', F.to_date(F.col('Date')))
  features = features.withColumn('Date', F.to_date(F.col('Date')))
  holidays = holidays.withColumn('Date', F.to_date(F.col('Date')))
  
  # Creating year_week column --> base for all date joins and comparisons & drop cols
  data = data.withColumn('year',F.year(F.col('Date')))\
      .withColumn('month',F.month(F.col('Date')))\
      .withColumn('weekofyear', F.weekofyear(F.col('Date')))\
      .withColumn('year_week', F.concat(F.col('year'),F.lit("_"),F.col('weekofyear')))\
      .withColumn('store_year_week', F.concat(F.col('Store'),F.lit("_"),F.col('year_week')))
  
  features = features.withColumn('year',F.year(F.col('Date')))\
      .withColumn('weekofyear', F.weekofyear(F.col('Date')))\
      .withColumn('year_week', F.concat(F.col('year'),F.lit("_"),F.col('weekofyear')))\
      .drop(*['Date','year','weekofyear','IsHoliday'])
  
  holidays = holidays.withColumn('year',F.year(F.col('Date')))\
                      .withColumn('weekofyear', F.weekofyear(F.col('Date')))\
                      .withColumn('year_week', F.concat(F.col('year'),F.lit("_"),F.col('weekofyear')))\
                      .drop(*['Date','year','weekofyear'])
  
  # Merging data
  X = data.join(stores, on= 'Store', how='left')\
      .join(features, on=['Store','year_week'], how='left')\
      .join(holidays, on = 'year_week',how='left').fillna(0)
  
  # Casting Cols
  X = X.withColumn('IsHoliday', F.col('IsHoliday').cast('int'))
  
  return X

full_train = preprocess_data(train,stores,features,holidays)
full_test = preprocess_data(test,stores,features,holidays)

# persisting data
full_train.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_full_train1')
full_test.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_full_test1')
````
In this step, all tables were consolidated into a single master table with column data types already casted. In this step
we also created some time related columns and included holidays flags.

## Modeling

In this step let's try different approaches in order to make our forecast predictions.

### Time Series Validation framework

The first thing we need to have in mind is our validation framework. A typical machine learning framework consists of
training a set of models or combination of model(s) on a training set and assessing its accuracy on holdout data or test
and validation sets (including also cross-validation frameworks). 

For forecasting, backtesting is the main tool to assess forecast accuracy. 

The time series forecasting characteristic time makes it different, in terms of evaluation
and backtesting methodology, from other fields of applied machine learning. Usually in
ML tasks, to assess the predictive error in a backtest, you split a data set by items. For
example, for cross-validation in image-related tasks, you train on some percentage of
the pictures, and then use other parts for testing and validation. In forecasting, you need
to split primarily by time (and to a lesser degree by items) to ensure that you do not leak
information from the training set into the test or validation set, and that you simulate the
production case as closely as possible. 

The split by time must be done carefully because you do not want to choose a single
point in time, but multiple points. Otherwise, the accuracy is too dependent on the
forecast start date, as defined by the splitting point. 

#### Roling Forecasting Evaluation

A time rolling window is an appropriate choice for a time series evaluation.

A rolling forecast evaluation, where
you do a series of splits over multiple time points, and output the average result leads to
more robust and reliable backtest results

![](/assets/ts/16.png)

**Importance of a rolling window evaluation**

The reason that multiple backtest windows are needed is that most time series in the
real world are normally non-stationary. By having multiple backtest windows, you can evaluate forecasting 
models in a more balanced and robust setting.

![](/assets/ts/17.png)

In our example, let's use a 2 period rolling window validation to train our models and average the result. 

So, let's create and persist our validation datasets.

````python

val1_date = datetime.datetime(2011,9,19)
val2_date = datetime.datetime(2012,3,16)

# Creating Validation Folders
cv_train_1 = full_train.where(F.col('Date')<=val1_date)
cv_fold_1 = full_train.where((F.col('Date')> val1_date)&
                             (F.col('Date')<= val2_date))

cv_train_2 = full_train.where(F.col('Date')<=val2_date)
cv_fold_2 = full_train.where(F.col('Date')>val2_date)

# Persisting Data
cv_train_1.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_cv_train_1')
cv_train_2.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_cv_train_2')
cv_fold_1.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_cv_fold_1')
cv_fold_2.write.format('parquet').mode('overwrite').save('dbfs:/FileStore/tables/walmart_cv_fold_2')

````

### H2O for Machine Learning

Basic ML Lifecycle 

![](/assets/ts/18.png)

H2O allow users to use 2 main frameworks: algorithms or FE + algorithms (h2o driverless AI).

![](/assets/ts/19.png)

H2O combined with Spark for ML Lifecycles:

![](/assets/ts/20.png)

#### Baseline Model

Let's create our h2o ML Baseline Model in our Databricks environment. 
It'll be a GAM model: generalized additive model (GAM) -> generalized linear model

**Note: H2O Installation databricks**

```` h2o_pysparkling_3.0 ```` : when creating our cluster we can use the Libraries installation option, through the
PyPI Source to enter this specific package.

````python
from pysparkling import *
hc = H2OContext.getOrCreate()

from pysparkling.ml import H2OGAMClassifier
from pysparkling.ml import H2ODRF
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
import numpy as np

import datetime

train_file = 'dbfs:/FileStore/tables/walmart_full_train1'
test_file = 'dbfs:/FileStore/tables/walmart_full_test1'
cv_train_1_file = 'dbfs:/FileStore/tables/walmart_cv_train_1'
cv_train_2_file = 'dbfs:/FileStore/tables/walmart_cv_train_2'
cv_fold_1_file = 'dbfs:/FileStore/tables/walmart_cv_fold_1'
cv_fold_2_file = 'dbfs:/FileStore/tables/walmart_cv_fold_2'

# Loading Data
full_train = spark.read.format('parquet').load(train_file)
full_test = spark.read.format('parquet').load(test_file)
cv_train_1 = spark.read.format('parquet').load(cv_train_1_file)
cv_train_2 = spark.read.format('parquet').load(cv_train_2_file)
cv_fold_1 = spark.read.format('parquet').load(cv_fold_1_file)
cv_fold_2 = spark.read.format('parquet').load(cv_fold_2_file)

from pysparkling.ml import H2ODRFRegressor

features = ['Store', 'Dept','IsHoliday','month', 'weekofyear','Type', 'Size','Temperature', 'Fuel_Price',
           'CPI', 'Unemployment', 'fl_superbowl', 'fl_tksgiving', 'fl_xtmas', 'fl_laborday']
cols_to_ignore = [col for col in cv_train_1.columns if col not in features]
target = 'Weekly_Sales'

estimator = H2ODRFRegressor(featuresCols=features, labelCol = target)
model = estimator.fit(full_train)

predictions = model.transform(full_test)
submission = predictions.withColumn('Id',
                                   F.concat(F.col('Store'),F.lit('_'),
                                           F.col('Dept'),F.lit('_'),
                                           F.col('Date')))\
            .withColumnRenamed('prediction','Weekly_Sales')\
            .select('Id','Weekly_Sales')

display(submission)
````
![](/assets/ts/21.png)

This Distributed Random Forest Regressor presented a score of 5170.34305

Now let's check the baseline XGBoost Performance:

````python
from pysparkling.ml import H2OXGBoostRegressor

features = ['Store', 'Dept','IsHoliday','month', 'weekofyear','Type', 'Size','Temperature', 'Fuel_Price',
           'CPI', 'Unemployment', 'fl_superbowl', 'fl_tksgiving', 'fl_xtmas', 'fl_laborday']
cols_to_ignore = [col for col in cv_train_1.columns if col not in features]
target = 'Weekly_Sales'

estimator = H2OXGBoostRegressor(featuresCols=features, labelCol = target)

model = estimator.fit(full_train)

predictions = model.transform(full_test)
submission = predictions.withColumn('Id',
                                   F.concat(F.col('Store'),F.lit('_'),
                                           F.col('Dept'),F.lit('_'),
                                           F.col('Date')))\
            .withColumnRenamed('prediction','Weekly_Sales')\
            .select('Id','Weekly_Sales')
````
THe baseline XGBoost presented a 6527.89845 score. 

## Testing some FE and Validation Framework




