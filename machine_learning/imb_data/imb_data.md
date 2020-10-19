# Imbalanced Datasets

In many real case scenarios, data is often imbalanced: the distribution between classes in your dataset is not equal. Imbalanced data 
usually refers to classification problems where classes are not equally distributed. 

Imbalanced data can occur for both binary and multiclasses classification problems. 

Some problems are expected to be imbalanced by its own nature. Examples: Fraud detection, Churn, etc.

**Why to handle imbalanced classes distributions?**

In many cases, it's possible to improve our ml metrics by handling imbalanced datasets accross several ways. Let's explore 
some of those techniques in this section.

## How to handle imbalanced datasets?

Let's go through some approaches to handle this situation.

### 1. Collect more data

If possible, we need to collect more data. A larger dataset will bring more cases (especially for the less frequent
classes) and this will help our models to learn. This process can also help other steps in this process (when resampling
the dataset, for example). If we're able to collect more negative class (assumed to be the less frequent) whe can later
undersample the full data with more data for class 1.

### 2. Metric

When dealing with imbalanced datasets, it's crucial to use a good metric to evaluate our problems. **There are metrics
that are more indicated to be used with imbalanced data**.

A good reference for metric selection can be found here: https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/

![](/assets/ml/imb_data/1.png)

In general terms, the ROC_AUC score can be used for most of imbalanced dataset problems. But it could also be helpful
to look other metrics, like the KAPPA score. 

> Kappa (or Cohen’s kappa)
- Classification accuracy normalized by the imbalance of the classes in the data.

> ROC Curves 
- Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.

#### Accuracy Paradox

Accuracy should almost never be used when we're dealing with imbalanced datasets. 

A classic example about the accuracy paradox occurs when we have highly imbalanced datasets, let's say 95% of class 0 and
5% of class 1. In this situation, we could create a naive classifier that will always output class 0 for a given observation
regardless of its features. 

If we use this naive model, we would obtain a 95% accuracy score. Just because of the data distribution and this could
be a **misleading** interpretation of the model's real performance. 

### 3. Resample Data

Here, we change our raw dataset trying to help our algorithms to improve the learning process.

Several resampling techniques can be applied in this step but generally they fall under 2 categories:

> Undersampling
- We reduce the number of the over-represented class.

> Oversampling
- We increase the number of the under-represented class.
- Here a popular technique is the **SMOTE**, when we create synthetical data (less frequent class) to over sample our dataset)

**When to choose undersampling/oversampling ?**

Generally we can use these rules of thumb:
* Consider testing under-sampling when you have an a lot data (tens- or hundreds of thousands of instances or more)
* Consider testing over-sampling when you don’t have a lot of data (tens of thousands of records or less)
* Consider testing random and non-random (e.g. stratified) sampling schemes.
* Consider testing different resampled ratios (e.g. you don’t have to target a 1:1 ratio in a binary classification problem, try other ratios)

#### Probability Calibration

Especially when treating imbalanced datasets with resampling techniques, we should always pay attention to the 
probability calibration step. If we need to use probabilities (to create a business case or estimate real applications
of the models), it's fundamental to calibrate our outputs probabilities. This can be done through the usage of specific
techniques like:
* Platt Scaling
* Isotonic Regression

### 4.Algorithms

It's always a good idea to test several algorithms when modeling a problem. Tree-based algorithms are usually able handle imbalanced
data with an acceptable performance. 

#### Penalizing models

We can also use penalties when training our models. Some algorithms accept **weights** as parameters when dealing with 
imbalance datasets. 

## Case study

In this section, let's test and compare different approaches to imbalanced datasets. 

We'll be comparing different datasets, metrics, algorithm, resampling techinques in order to check results.

Dataset: https://www.kaggle.com/ntnu-testimon/paysim1
- It contains 6362620 entries with 11 Columns. 

````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

SEED = 42

# Load initial data
data = pd.read_csv('pr_log.csv')
# Full resample of data
data = data.sample(frac=1,random_state=SEED).reset_index(drop=True)

# Train & Test split
test_ratio = 0.3
test_cutoff_index = int(len(data)*test_ratio)
train = data[:test_cutoff_index]
test = data[test_cutoff_index:]

data.head()
```` 
![](/assets/ml/imb_data/2.png)

Let's check the distribution of our target variable
````python
target = 'isFraud'
train[target].value_counts(normalize=True)
````
![](/assets/ml/imb_data/3.png)

As we can see, we're dealing with a highly imbalanced dataset (class 1: 0.12%)

Let's create our validation framework and baseline models.

````python
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import transformers
from sklearn.model_selection import cross_val_score

target = 'isFraud'
numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
       'oldbalanceDest', 'newbalanceDest']

categorical_cols = ['type']
features = numerical_cols + categorical_cols

def get_cv_scores(data, model, num_cols, cat_cols, target, nfolds,seed,
                 resampling=None,scoring='roc_auc'):
    
    # CV object
    SKfold = StratifiedKFold(n_splits = nfolds, random_state=seed)
    # Pipeline
    pipe = Pipeline([
            # Treating New Categories
            ("Unseen Labels", transformers.TreatNewLabels(variables=cat_cols)),
            # Numerical Missing Imputation
            ("Value Imputer", transformers.NumericalNAImputerValue(variables=num_cols, add_column=False)),
            # Categorical Missing Imputation
            ("Categorical Imputer Label", transformers.CategoricalImputerLabel(variables=cat_cols)),
            # Categorical Encoding --> OHE
            ("Categorical Encoding OHE", transformers.CategoricalEncoderOHE()),
            # Scaling
            ("Scaling", transformers.AdjustedScaler()),
            # Model
            ("model", model)
        ])
    # getting cv scores
    features = num_cols + cat_cols
    X_cv = data[features]
    y_cv = data[target]
    scores = cross_val_score(pipe, X=X_cv, y=y_cv, 
                             scoring=scoring, 
                             cv=SKfold, n_jobs=-1)
        
    return scores

from sklearn.linear_model import LogisticRegression

# Baseline Model --> Logistic Regression
result_logreg = get_cv_scores(data=train.sample(frac=0.15),
                              model=LogisticRegression(),
                              num_cols=numerical_cols, 
                              cat_cols=categorical_cols,
                              target=target,
                              nfolds=5,
                              seed=SEED,
                              resampling=None,
                              scoring='roc_auc')

print(result_logreg)
print(f"Avg result: {round(np.mean(result_logreg),4)}")
>> [0.97571035 0.9670854  0.96619548 0.97011154 0.93886533]
>> Avg result: 0.9636
````

How other models (default behavior handle this imbalanced dataset?)

````python
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

models_dict = {'LGBM':LGBMClassifier(), 
               'XGBoost': XGBClassifier(),
               'RF': RandomForestClassifier(),
               'LR': LogisticRegression(),
              'KNN': KNeighborsClassifier(),
              'ExtraTree': ExtraTreesClassifier(),
              'DT': DecisionTreeClassifier()}

df_results = pd.DataFrame()
fold_1, fold_2, fold_3, fold_4, fold_5 = [],[],[],[],[]
model_names, metric_names = [],[]
avg_metrics = []
test_scores = []

sample_train = train.sample(frac=0.15,random_state=SEED)
X_train = sample_train[features]
y_train = sample_train[target]
X_test = test[features]
y_test = test[target]

# Test Results
fe_pipe = Pipeline([
    # Treating New Categories
    ("Unseen Labels", transformers.TreatNewLabels(variables=categorical_cols)),
    # Numerical Missing Imputation
    ("Value Imputer", transformers.NumericalNAImputerValue(variables=numerical_cols, add_column=False)),
    # Categorical Missing Imputation
    ("Categorical Imputer Label", transformers.CategoricalImputerLabel(variables=categorical_cols)),
    # Categorical Encoding --> OHE
    ("Categorical Encoding OHE", transformers.CategoricalEncoderOHE()),
    # Scaling
    ("Scaling", transformers.AdjustedScaler())
])
     
fe_pipe.fit(X_train,y_train)
X_train_mod = fe_pipe.transform(X_train)
X_test_mod = fe_pipe.transform(X_test)

for model_name in models_dict.keys():
    
    model_names.append(model_name)
    selected_model = models_dict[model_name]
    selected_scoring = 'roc_auc'
    
    model_results = get_cv_scores(data=sample_train,
                              model = selected_model,
                              num_cols=numerical_cols, 
                              cat_cols=categorical_cols,
                              target=target,
                              nfolds=5,
                              seed=SEED,
                              resampling=None,
                              scoring=selected_scoring)
    # CV Results
    fold_1.append(model_results[0])
    fold_2.append(model_results[1])
    fold_3.append(model_results[2])
    fold_4.append(model_results[3])
    fold_5.append(model_results[4])
    avg_metrics.append(np.mean(model_results))
    metric_names.append(selected_scoring)
    
    
    # Test Results
    full_model = models_dict[model_name]
    full_model.fit(X_train_mod,y_train)
    
    y_test_scores = full_model.predict_proba(X_test_mod)[:,1]
    test_roc_auc = roc_auc_score(y_test,y_test_scores)
    test_scores.append(test_roc_auc)

df_final_results = pd.DataFrame({"model_name":model_names,
                                "metric":metric_names,
                                "fold_1":fold_1,
                                "fold_2":fold_2,
                                "fold_3":fold_3,
                                "fold_4":fold_4,
                                "fold_5":fold_5,
                                "avg_cv_metric":avg_metrics,
                                "test_metric":test_scores})

df_final_results.sort_values('avg_metric', ascending=False,inplace=True)
df_final_results
````
![](/assets/ml/imb_data/5.png)

### Resampling Techniques

For resampling techniques let's use the imblearn library.

SMOTE-NC can be applied to categorical variables but need to pay attention to categorical pre_processing first. 
Especially with new labels in test. 

````python
import imblearn

n_folds = 5
SEED = 42

cv = StratifiedKFold(n_splits=n_folds, random_state=SEED, shuffle=True)

sample_train = train.sample(frac=0.15,random_state=SEED).reset_index(drop=True)

X = sample_train[features]
y = sample_train[target]

X_test = test[features]
y_test = test[target]


models_dict = {'LGBM':LGBMClassifier(), 
               'XGBoost': XGBClassifier(),
               'RF': RandomForestClassifier(),
               'LR': LogisticRegression(),
              'KNN': KNeighborsClassifier(),
              'ExtraTree': ExtraTreesClassifier(),
              'DT': DecisionTreeClassifier()}

resampling_dict = {'No Resampling':None,
                    'RandomUnderSampling': imblearn.under_sampling.RandomUnderSampler(random_state=SEED),
                   #'SMOTENC':imblearn.over_sampling.SMOTENC(categorical_features=categorical_cols,
                   #                                         random_state=SEED),
                   'RandomOversampling':imblearn.over_sampling.RandomOverSampler(random_state=SEED)}

resampling_names = []
model_names = []
validation_scores = []
test_scores = []

for resampler_name in resampling_dict.keys():
    #print(f"Resamplet name {resampler_name}")
    for model_name in models_dict.keys():
        #print(f"Model name {model_name}")
        folds_scores = []
        for i, (train_index, val_index) in enumerate(cv.split(X=X, y=y)):
            #print(f"Fold {i}")
            # Creating Train and Validation under the CV framework
            df_train, df_validation = sample_train.loc[train_index], sample_train.loc[val_index]
            X_train, y_train = df_train[features], df_train[target]
            X_validation, y_validation = df_validation[features], df_validation[target]
            
            if resampler_name != 'No Resampling':
                # For each fold > Resamples > FE Pipe > Model > Val Score
                resampler = resampling_dict[resampler_name]
                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
                
            # FE Pipeline 
            fe_pipe = Pipeline([
                # Treating New Categories
                ("Unseen Labels", transformers.TreatNewLabels(variables=categorical_cols)),
                # Numerical Missing Imputation
                ("Value Imputer", transformers.NumericalNAImputerValue(variables=numerical_cols, add_column=False)),
                # Categorical Missing Imputation
                ("Categorical Imputer Label", transformers.CategoricalImputerLabel(variables=categorical_cols)),
                # Categorical Encoding --> OHE
                ("Categorical Encoding OHE", transformers.CategoricalEncoderOHE()),
                # Scaling
                ("Scaling", transformers.AdjustedScaler())])

            fe_pipe.fit(X_train_resampled, y_train_resampled)
            new_X_train = fe_pipe.transform(X_train_resampled)
            new_X_validation = fe_pipe.transform(X_validation)

            selected_model = models_dict[model_name]
            selected_model.fit(new_X_train, y_train_resampled)

            # Validation Scores
            y_val_scores = selected_model.predict_proba(new_X_validation)[:,1]
            val_rocauc = roc_auc_score(y_validation, y_val_scores)
            folds_scores.append(val_rocauc)
        
        avg_val_scores = np.mean(folds_scores)
        validation_scores.append(avg_val_scores)
        resampling_names.append(resampler_name)
        model_names.append(model_name)

        # Test Results
        if resampler_name != 'No Resampling':
            # For each fold > Resamples > FE Pipe > Model > Val Score
            resampler = resampling_dict[resampler_name]
            X_resampled, y_resampled = resampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        # FE Pipeline 
        fe_pipe = Pipeline([
            # Treating New Categories
            ("Unseen Labels", transformers.TreatNewLabels(variables=categorical_cols)),
            # Numerical Missing Imputation
            ("Value Imputer", transformers.NumericalNAImputerValue(variables=numerical_cols, add_column=False)),
            # Categorical Missing Imputation
            ("Categorical Imputer Label", transformers.CategoricalImputerLabel(variables=categorical_cols)),
            # Categorical Encoding --> OHE
            ("Categorical Encoding OHE", transformers.CategoricalEncoderOHE()),
            # Scaling
            ("Scaling", transformers.AdjustedScaler())])

        fe_pipe.fit(X_resampled, y_resampled)
        new_X_resampled = fe_pipe.transform(X_resampled)
        new_X_test = fe_pipe.transform(X_test)
        
        selected_model = models_dict[model_name]
        selected_model.fit(new_X_resampled, y_resampled)
        # Test Scores
        y_test_scores = selected_model.predict_proba(new_X_test)[:,1]
        test_rocauc = roc_auc_score(y_test, y_test_scores)
        test_scores.append(test_rocauc)

df_results = pd.DataFrame({"model":model_names,
                          "resampler":resampling_names,
                          "cv_avg_metric":validation_scores,
                          "test_metric":test_scores})
df_results
````

![](/assets/ml/imb_data/6.png)

![](/assets/ml/imb_data/7.png)

![](/assets/ml/imb_data/8.png)

