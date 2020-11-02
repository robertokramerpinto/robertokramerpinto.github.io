# Sparkling Water

Sparkling Water allows users to combine the fast, scalable machine learning algorithms of H2O with the capabilities of Spark. 
With Sparkling Water, users can drive computation from Scala/R/Python and utilize the H2O Flow UI, providing an ideal 
machine learning platform for application developers.

H2O combined with Spark for ML Lifecycles typically involves data munging with the help of a Spark API, where a
prepared table is passed to the H2O ML algorithm. All FE process must be made in Spark. H2O will handle only the pure ML part of the framework.

![](assets/ts/20.png)

Let's see some examples on how to implement h2o pysparkling algos. 

> Note: H2O Installation databricks
* ````h2o_pysparkling_3.0````
    * when creating our cluster we can use the Libraries installation option, through the PyPI Source to enter this specific package.


## Binary Classification example

Let's check a binary classification example implemented with pysparkling water algorithms. 

### H2ODRFClassifier - Distributed Random Forest 

````python
from pysparkling import *
hc = H2OContext.getOrCreate()

Loading CV Data
cv_data = spark.read.format('parquet').load('dbfs:/FileStore/tables/xsell_cv_data')

from pysparkling.ml import H2ODRFClassifier

selected_features = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 
'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
categorical_cols = ['Gender','Vehicle_Age','Vehicle_Damage']
target = 'Response'

estimator = H2ODRFClassifier(featuresCols=selected_features, labelCol=target,
                         columnsToCategorical = categorical_cols,
                         convertInvalidNumbersToNa=True,
                         convertUnknownCategoricalLevelsToNa=True,
                         foldCol='kfold')

model = estimator.fit(cv_data)

# Model Outputs
model.getModelDetails()
model.getCrossValidationMetrics()

# Estimation
model.transform(test_df).show(truncate = False)

````
**Notes**

> Missing values
* H2O considers by default missing values as information
* Because of that it considers it an independent value to be used in the Dataset
* No imputation is performed

### H2OXGBoostClassifier

````python
from pysparkling.ml import H2OXGBoostClassifier

selected_features = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
categorical_cols = ['Gender','Vehicle_Age','Vehicle_Damage']
target = 'Response'

estimator = H2OXGBoostClassifier(featuresCols=selected_features, labelCol=target,
                                 columnsToCategorical = categorical_cols,
                                 convertInvalidNumbersToNa=True,
                                 convertUnknownCategoricalLevelsToNa=True,
                                 foldCol = 'kfold')

model = estimator.fit(cv_data)
model.getCrossValidationMetrics()
````

### Target Encoding with Pysparkling Water and XGBoost

Obs: applyting target enconding to entire dataset, not in the inner loop of cross validation as expected.

````python
# TargetEncoding with XGBoost 
from pysparkling.ml import H2OTargetEncoder
from pysparkling.ml import H2OXGBoostClassifier
from pyspark.ml import Pipeline

selected_features = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
categorical_cols = ['Gender','Vehicle_Age','Vehicle_Damage']
numerical_cols = [col for col in selected_features if col not in categorical_cols]
target = 'Response'

targetEncoder = H2OTargetEncoder()\
  .setInputCols(categorical_cols)\
  .setLabelCol(target)

encoder = targetEncoder.fit(cv_data)

preprocessed_cv_data = encoder.transform(cv_data)

te_categorical_cols = [col+"_te" for col in categorical_cols]
final_features = numerical_cols + te_categorical_cols

estimator = H2OXGBoostClassifier(featuresCols=final_features, labelCol=target,
                                 convertInvalidNumbersToNa=True,
                                 convertUnknownCategoricalLevelsToNa=True,
                                 foldCol = 'kfold')
model = estimator.fit(preprocessed_cv_data)
model.getCrossValidationMetrics()

````


