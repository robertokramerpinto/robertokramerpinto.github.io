

## CV framework

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



