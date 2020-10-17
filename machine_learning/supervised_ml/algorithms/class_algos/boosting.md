# Boosting Algorithms

## XGBoost Classifier

## LightGBM Classifier 

LightGBM is a gradient boosting framework that uses tree based learning algorithms. 

It's designed to achieve the following advantages:
- Faster training and higher efficiency
- Lower memory usage
- Better accuracy
- Handle large data

### Python implementation 

```python
!pip install lightgbm
import lightgbm as lgb
```

Let's start by importing general python libraries and loading initial data
````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb


data = pd.read_csv("data.csv")
data.head().T
````

![](/assets/ml/supervised/algos/1.png)

The dataset that we'll be using for this example is the Heart Disease dataset (DriveData competition).

Let's now split our data into train and test sets.

````python
target = "heart_disease_present"
features = [col for col in data.columns if col not in [target]]
y = data[target]
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

````

Implementing Model Only

````python
# LightGBM Classifier

model = lgb.LGBMClassifier()
model.fit(X_train,y_train)

# Train Scores
y_train_scores = model.predict_proba(X_train)[:,1]
# Test Scores
y_test_scores = model.predict_proba(X_test)[:,1]

# Performance 
auc_train = roc_auc_score(y_train,y_train_scores)
auc_test = roc_auc_score(y_test,y_test_scores)

print(f"ROC-AUC Train: {round(auc_train,4)}")
print(f"ROC-AUC Test: {round(auc_test,4)}")
>> ROC-AUC Train: 0.999
>> ROC-AUC Test: 0.8395
````

### Hyperparameters

> Num_leaves
- Maximum tree leaves for base learners.
> max_depth (int, optional (default=-1))
- Maximum tree depth for base learners, <=0 means no limit.
> learning_rate (float, optional (default=0.1))
- Boosting learning rate. 
> n_estimators (int, optional (default=100))
- Number of boosted trees to fit.
> objective (string, callable or None, optional (default=None))
- Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below). Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.
> class_weight (dict, 'balanced' or None, optional (default=None))
- Weights associated with classes in the form {class_label: weight}. Use this parameter only for multi-class classification task; for binary classification task you may use is_unbalance or scale_pos_weight parameters. 
The ‘balanced’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
> min_split_gain (float, optional (default=0.))
- Minimum loss reduction required to make a further partition on a leaf node of the tree.
> min_child_weight (float, optional (default=1e-3))
- Minimum sum of instance weight (hessian) needed in a child (leaf).
> min_child_samples (int, optional (default=20))
- Minimum number of data needed in a child (leaf).
> subsample (float, optional (default=1.))
- Subsample ratio of the training instance.
> subsample_freq (int, optional (default=0))
- Frequence of subsample, <=0 means no enable.
> colsample_bytree (float, optional (default=1.))
- Subsample ratio of columns when constructing each tree.
> reg_alpha (float, optional (default=0.))
- L1 regularization term on weights.
> reg_lambda (float, optional (default=0.))
- L2 regularization term on weights.
> random_state
 
> n_jobs (int, optional (default=-1)) – Number of parallel threads.

> importance_type (string, optional (default='split'))
- The type of feature importance to be filled into feature_importances_. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.

**Control OverFitting**

- Use small ```max_bin```
- Use small ```num_leaves```
- Use ```min_data_in_leaf``` and ```min_sum_hessian_in_leaf```
- Use bagging by set ```bagging_fraction``` and ```bagging_freq```
- Use feature sub-sampling by set ```feature_fraction```
- Use bigger training data
- Try ```lambda_l1```, ```lambda_l2``` and ```min_gain_to_split``` for regularization
- Try ```max_depth``` to avoid growing deep tree
- Try ```extra_trees```
- Try increasing ```path_smooth```

````python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'learning_rate': [0.1,0.05],
    'n_estimators': [200,300,400],
    'min_child_samples':[20,30,50],
    'max_depth':[7,10,15,-1],
    'num_leaves': [30,50,100,200,300],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [42], # Updated from 'seed'
    'colsample_bytree' : [0.9,1],
    'subsample' : [0.9,1],
    #'reg_alpha' : [1,1.5],
    #'reg_lambda' : [1,1.5],
    "n_jobs":[-1]
    }

# Create a based model
model = lgb.LGBMClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X,y)

# Best Model
best_model = grid_search.best_estimator_
optimal_parameters = grid_search.best_params_

#CV best results
print(f"CV Best scores: {round(grid_search.best_score_,4)}")
>>  CV Best scores: 0.8278
````

### Model Outputs

**Feature Importance**

````python
_ = lgb.plot_importance(model, height=0.3, figsize=(10,6))
````
![](/assets/ml/supervised/algos/5.png)




## Catboost 


