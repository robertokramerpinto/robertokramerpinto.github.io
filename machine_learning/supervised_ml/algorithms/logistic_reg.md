# Logistic Regression

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

## Sklearn Implementation

Let's start by importing general python libraries and loading initial data
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

### Hyperparameters

* C
    * Inverse of the regularization strenght. 
    * The higher the C value, the lower the regularization

* Penalty
    * Indicates the type of regularization
    * l1 (LASSO), l2 (Ridge)

* solver
    * algorithm used in the optimization problem
    * Some solvers do not accept all regularization types

* Class weight
    * weights associated with classes in the form {label:weight}
    * Default -> 1
    * accepts dict or 'balanced'; Balanced automatically adjusts the weights invers. proportional to class freq. (n_samples / (n_classes * np.bincount(y)))

````python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    "C": [0.001, 0.01, 0.5, 0.1],
    "penalty":["l1","l2"],
    "random_state": [42],
    "max_iter":[1000],
    "n_jobs":[-1]
}

# Create a based model
model = LogisticRegression()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X,y)

# Best Model
best_model = grid_search.best_estimator_
optimal_parameters = grid_search.best_params_

````

### Model Outputs

**Coefficients**

````python
# examine the coefficients
logreg.coef_
# Examine the intercept 
logreg.intercept_
````

> How to interpret logistic regression coefficients?
- A 1 unit increase in a given variable is associated with a <coef_1> unit increase/decrease in the __log-odds__ of the
target.

Positive coefficients increase the __log-odds__ of the response (and thus increase the probability) and negative 
coefficients decrease the __log-odds__ of the response variable (and thus decrease the probability).

The target is the baseline log-odds level for the response (if all variables are null).

![](/assets/ml/supervised/algos/4.png)


