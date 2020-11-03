# Machine Learning & Spark 

PySpark for ML problems comes in hand whenever we have large sets of data. If data no longer fits into memory, then we
have a good candidate for Pyspark ML applications. 

PySpark handles this issue by distributing the problem across multiple computers in a cluster. Data is divided into
partitions (groups of data). Ideally, each partition can fit into RAM on a single computer in the cluster. 

![](/assets/pyspark/1.png)

* Spark is a general distributed data processing engine
* Computations are performed across a distributed cluster of computers
* Data is processed in memory
* Well documented high-level API

![](/assets/pyspark/2.png)

## PySpark 

To use spark with python we can use the pyspark package.

In addition to pyspark we also have other very useful submodules:

> pyspark.sql
* To work with structured data
* DataFrame objects

> pyspark.streaming
* To work with streaming data

> pypsark.ml 
* To process machine learning flows

Let's now focus on how to use PySpark to perform ML flows, addressing topics such as:
* Feature Engineering
* Pipelines
* Algorithms
* Models Evaluation
* Deployment

## Feature Engineering 

### Missing Values

Currently, Spark can only handle numerical imputing missing values. 
As indicated it's best to use a preprocessing function outside pipeline flow to handle missing values.

### Handling Categorical Data

Pyspark.ml currently accepts 2 methods for handling categorical features: StringIndex and OHE. 

> String Index
* StringIndexer encodes a string column of labels to a column of label indices.
* If the input column is numeric, we cast it to string and index the string values.

> OHE (One Hot Encoding)
* One-hot encoding maps a categorical feature, represented as a label index, to a binary vector with at most a 
single one-value indicating the presence of a specific feature value from among the set of all feature values. 
This encoding allows algorithms which expect continuous features, such as Logistic Regression, 
to use categorical features. For string type input data, it is common to encode categorical features using **StringIndexer first**.

## Regression Case Example

````python
import pyspark
from pyspark.sql import udf
import pyspark.sql.functions as F
import pyspark.sql.types as T

import pandas as pd

# Loading data
flights = spark.sql("select * from flights")

# Replacing NA by nulls
fill_nulls_udf = F.udf(lambda x: None if x in ['NA'] else x, T.StringType())
for col in ['carrier','org','delay']:
  flights = flights.withColumn(col, fill_nulls_udf(F.col(col)))
flights = flights.na.drop(subset='delay')

# FE Process 

def preprocess_data(data, target_label='target',num_features=[],cat_features=[],
                   num_missing_value=-1, cat_missing_value='missing'):
  
  # Treating Missing Values
  X = data
  X = X.na.fill(value=num_missing_value, subset = num_features)
  X = X.na.fill(value=cat_missing_value, subset = cat_features)
  
  # Casting target col
  X = X.withColumn(target_label, F.col(target_label).cast('int'))
  
  return X

target_label = 'delay'
num_features = ['mon', 'dom', 'dow','flight','mile', 'depart', 'duration']
cat_features = ['carrier', 'org']

preprocessed_train = preprocess_data(data = train, target_label = target_label,
                                    num_features=num_features, cat_features=cat_features)

preprocessed_test = preprocess_data(data = test, target_label = target_label,
                                    num_features=num_features, cat_features=cat_features)

# Pipeline Categorical Features
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler

# -- Stage 1 : String Index (categorical cols)
stage_1 = StringIndexer(inputCols=cat_features, outputCols=[c+"_index" for c in cat_features], handleInvalid='keep')
# -- Stage 2: OHE (categorical cols --> String Indexer)
stage_2 = OneHotEncoder(inputCols=stage_1.getOutputCols(), outputCols=[c+"_ohe" for c in stage_1.getOutputCols()])
# -- Stage 3: Vector Assembler
stage_3 = VectorAssembler(inputCols= num_features + stage_2.getOutputCols(),
                         outputCol = 'features')
# -- Stage 4: Scaling
stage_4 = StandardScaler(inputCol='features', outputCol='scaled_features')

pipeline = Pipeline(stages=[stage_1, stage_2, stage_3, stage_4])
pipeline_model = pipeline.fit(preprocessed_train)

# Transforming data
premodel_train = pipeline_model.transform(preprocessed_train)

# Save Pipeline
fe_pipe_path = 'dbfs:/FileStore/pipelines/flights_fe_pipe'
# Save model 
pipeline_model.write().overwrite().save(fe_pipe_path)

# Loading pipeline and transforming test data
from pyspark.ml.pipeline import PipelineModel
fe_pipe_path = 'dbfs:/FileStore/pipelines/flights_fe_pipe'
loaded_pipeline_model = PipelineModel.load(fe_pipe_path)

# Example - Linear Regression
from pyspark.ml.regression import LinearRegression

algo = LinearRegression(featuresCol="scaled_features", labelCol="delay")
# train the model
model = algo.fit(premodel_train)

# evaluation
evaluation_summary = model.evaluate(premodel_test)
# predicting values
predictions = model.transform(premodel_test)

print(f"MAE: {evaluation_summary.meanAbsoluteError}")
print(f"RMSE: {evaluation_summary.rootMeanSquaredError}")
print(f"R^2: {evaluation_summary.r2}")
>>MAE: 35.0948615547335
>>RMSE: 52.13430185931525
>>R^2: 0.0556421558393182
````

Cross validation Example for a simple regression model

````python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from pyspark.ml.evaluation import RegressionEvaluator

# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(featuresCol="scaled_features", labelCol="delay")
evaluator = RegressionEvaluator(labelCol="delay")

# Create a cross validator
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5, seed=42)

# Train and test model on multiple folds of the training data
cv = cv.fit(premodel_train)

# Average metrics
print(f"Avg. RMSE (folds): {cv.avgMetrics}")
>> Avg. RMSE (folds): [52.71280766779186]

# Make predictions based on model
evaluator.evaluate(cv.transform(premodel_test))
>> 52.13430185931525
````

Also possible to use a pipeline into the CV object. Example:
````python
# Create an indexer for the org field
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=['km', 'org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator)
````

Let's Create a full pipeline now full a random forest regressor. In this example, we'll be using the GridSearch with
CV approach. 

````python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor

# Create objects for building and evaluating a regression model
evaluator = RegressionEvaluator(labelCol="delay")
estimator = RandomForestRegressor(featuresCol="scaled_features", labelCol="delay")

# Create a parameter grid builder
params = ParamGridBuilder()\
     .addGrid(estimator.maxDepth, [2, 5, 10])\
     .build()

print(f"Number of Tested models: {len(params)}")
>> Number of Tested models: 3
   
# Create a cross validator
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5, seed=42)

# Train and test model on multiple folds of the training data
cv = cv.fit(premodel_train)

# Best Model
cv.bestModel

# Average RMSE for each parameter combination in grid
print(f"Avg metrics: {cv.avgMetrics}")

# Full CV Summary : metrics x validation
def extract_full_cv_results(cvModel):
  
  params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]
  cv_summary = pd.DataFrame.from_dict([
      {cvModel.getEvaluator().getMetricName(): metric, **ps} 
      for ps, metric in zip(params, cvModel.avgMetrics)
  ])
  return cv_summary

cv_summary = extract_full_cv_results(cv)
cv_summary

````







