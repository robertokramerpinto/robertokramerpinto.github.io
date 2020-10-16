# Supervised Machine Learning

Let's explorer some binary classification algorithms:

> KNN

> Naive Bayes

> [Logistic Regression](#logistic-regression)

> Decision Trees

> Random Forest

> Boosting Algorithms
- XGBoost Classifier 
- LGBM Classifier
- CatBoost Classifier 

# Code Part

Let's start by importing general python libraries and load initial data
````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split


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

For simplicity reasons, the selected dataset only contains numerical features so we can skip the categorical encoding
step in this case. 

## Logistic Regression

Logistic Regression is a linear model used for classification. It's a regression method that outputs the probability 
of a binary target variable based on a given feature space. The logistic regression models the log-probability (belong to 
a given class).

The sklearn Logistic Regression model can be applied to both binary and multiclass classification problems. 

**Linear x Logistic Models**

The logit fuction transforms the linear regression into a smoother continuous output, bounded by 2 fixed limits (0 & 1).
- Logit Function will take any continuous input and output a value [0,1]
- This output can also be interpreted as a score and/or probability of an observation belonging to class 1

![](/assets/ml/supervised/algos/2.png)

**Notes**
- It's a good baseline model
- It's generally stable and present lower overfitting levels 

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

# Train Scores
y_train_scores = logreg.predict_proba(X_train)[:,1]
# Test Scores
y_test_scores = logreg.predict_proba(X_test)[:,1]

# Performance 
auc_train = roc_auc_score(y_train,y_train_scores)
auc_test = roc_auc_score(y_test,y_test_scores)

print(f"ROC-AUC Train: {round(auc_train,4)}")
print(f"ROC-AUC Test: {round(auc_test,4)}")
>>ROC-AUC Train: 0.9175
>>ROC-AUC Test: 0.8653
```

