# SHAP Values

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of **any machine learning model**.
It uses the classic Shapley values from game theory and their related extensions. 

* Installation

````python
pip install shap

conda install -c conda-forge shap
````

## Explainable AI with SHAP Values

### Additive Nature of SHAP values

One fundamental property of Shapley values is that they always sum up to the difference between the game outcome when all 
players are present and the game outcome when no players are present. 

For machine learning models this means that SHAP values of all the input features will always sum up to the difference 
between baseline (expected) model output and the current model output for the prediction being explained.

The easiest way to see this is through a waterfall plot that starts our background prior expectation for a home 
price E[f(x)], and then adds features one at a time until we reach the current model output f(x):

![](/assets/ml/shap/5.png)

SHAP values are always referencing the difference between a given output and the expected value (baseline).

### Linear Regression Model 

A linear regression model is one of the simplest model types that we can use for regression problems. Let's see how SHAP
can be used in this case. We'll be using the Boston housing dataset (already available under the shap package).

This dataset consists of 506 neighboorhood regions around Boston in 1978, where our goal is to predict the median 
home price (in thousands) in each neighboorhood from 14 different features

````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#!pip install shap
import shap
import sklearn

# Loading Boston Dataset
X,y = shap.datasets.boston()

# Fitting a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# Extracting Coefficients
linear_model_coeff = pd.DataFrame({"feature":X.columns,
                                   "coefficient": model.coef_})
linear_model_coeff
````
![](/assets/ml/shap/1.png)

The traditional way to understand a linear model is to analyze its coefficients. These coefficients will tell us how much 
our model output will vary depending on the change of the input features. 

**Feature importance approach**

While individual coefficients are good to understand individual variations, they don't present us a great way to 
understand feature importance as whole. This is because each coefficient will depend on the feature's scale. 

If a coefficient is larger than other it might be because of the difference in scale between both features. This means that
the magnitude of a coefficient is not necessarily a good measure of a feature's importance in a linear model.

#### Partial dependence plots

Partial dependence plots try to understand how changing the feature impacts the model and its own distribution. Let's check 
this example.

````python
sample_df = X.sample(n=200, random_state=42)
shap.plots.partial_dependence("AGE", 
                              model.predict, 
                              sample_df, ice=False, 
                              model_expected_value=True, 
                              feature_expected_value=True,
                              ace_opacity=0,pd_opacity=1)
````
![](/assets/ml/shap/3.png)

For this linear model, the grey horizontal line represents the Expected value of the model when applied to the boston 
housing dataset. The grey vertical line represents the average value of the AGE feature. The blue line (dependence plot)
represents the average value of the model when we fix a given age value. 

The intersection between the lines is known as the "center" of the partial dependence plot with respect to the data
distribution. 

**SHAP Values**

The basic idea behind SHAP values is to use fair allocation results from cooperative game theory to allocate credit
for a model's output among its input features. 

ML & Game Theory:
* Map features to players
* Map model to game's rules

Let's compute SHAP values for the linear regression model.

```python
# compute the SHAP values for the linear model
background = shap.maskers.Independent(X, max_samples=1000)
explainer = shap.Explainer(model.predict, background)
shap_values = explainer(X)

shap.plots.scatter(shap_values[:,"AGE"])
```
![](/assets/ml/shap/4.png)

The SHAP value for a given feature represents the average shap values accross the entire dataset given the feature
distribution.

