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

data = pd.read_csv("data.csv")
data.head().T
````
![](/assets/ml/supervised/algos/1.png)

The dataset that we'll be using for this example is the Heart Disease dataset (DriveData competition).



## Logistic Regression

Logistic Regression
