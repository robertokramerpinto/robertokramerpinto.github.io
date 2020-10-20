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

![](/assets/ml/shap/7.png)

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
![](/assets/ml/shap/9.png)

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
![](/assets/ml/shap/10.png)

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

For each observation we can also understand the overall features contribution to the model's outcome compared to the
baseline output:

````python
# Individual contribution --> SHAP VALUES
sample_ind = 15
# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[sample_ind], max_display=14)

````
![](/assets/ml/shap/11.png)

## Linear Explainer

Let's use the linear Explainer to understand a Logistic Regression model.

Since we are explaining a logistic regression model the units of the SHAP values will be in the log-odds space.

````python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap

# print the JS visualization code to the notebook
shap.initjs()

X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Initiating and training Logistic Regression Model
model = sklearn.linear_model.LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Initiating SHAP Linear Explainer
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

shap_values_sum = [np.sum(i) for i in shap_values]
df_shap = pd.DataFrame({"total_shap":shap_values_sum,
                      "predicted_proba":y_test_scores})
df_shap
````
![](/assets/ml/shap/13.png)

The Shap value for each individual observation is the sum of each individual feature 
contribution. The values we see here is this df_shap dataframe are the relative difference 
to the expected shap value. 

Shap measures the impact of varaibles taking into consideration the interaction with other features. It's an interpretation of the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

The shap_values object is a list with two arrays.

The first array is the SHAP values for a negative outcome, and the second array is the list of SHAP values for the positive outcome (1).

It is important to keep in mind that XGBoost trees are built on the log-odds scale and then just transformed to probabilities for predict_proba. So SHAP values are also in log odds units.
A negative base value means you are more likely class 0 than 1, and the sum will equal the log-odds output of the model not the transformed probability after the logistic function.


```python
# Expected value for the given dataset
explainer.expected_value
>> -1.67
```
In our example, the expected value (baseline value) is -1.67.

The base value is the mean of the model output over the background dataset. 
So it depends what your model output is and what the background dataset is. 
For TreeExplainer the background dataset is typically the training dataset, 
and the model output depends on the type of problem (it is log-odds for XGBoost by default).

It's also the case for the Logistic Regression Model. 

**Summary the effect of all the features**
````python
#shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns,use_log_scale=True)
````
![](/assets/ml/shap/12.png)

This graph represents the general effect of each feature relative to the baseline value (0).

As we can see from the graph, hours per week feature presents (in general) a positive correlation with shap values: 
as the hours increase, shap values also increase.

**Individual plots**

It's also possible to understand individual observations, compared to the baseline. We can see, for each 
observation what are the major causes for the observed result.

````python
# -- Individual Observation
selected_index = 0
shap.force_plot(explainer.expected_value, shap_values[selected_index,:], X_test.iloc[selected_index,:],
                text_rotation=90,matplotlib=True)

# -- Individual Observation -- Probability
selected_index = 0
shap.force_plot(explainer.expected_value, shap_values[selected_index,:], X_test.iloc[selected_index,:],
                text_rotation=90,matplotlib=True,link='logit')
````
![](/assets/ml/shap/14.png)

The baseline value in the "probability" value is 0.1575 (probability of class 1). In this example, we can 
see that observation 0 presents predict_proba = 0.08 and through the force_plot we can explore
all factors that contribute relatively to the baseline to decrease the probability.

## XGboost Implementation

````python
from sklearn.model_selection import train_test_split
import xgboost
import shap
import numpy as np
import matplotlib.pylab as pl

# print the JS visualization code to the notebook
shap.initjs()

X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

````

### Exploring XGBoost Feature Importance

````python
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.show()

xgboost.plot_importance(model, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
pl.show()

xgboost.plot_importance(model, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
pl.show()

````
![](/assets/ml/shap/15.png)
![](/assets/ml/shap/16.png)
![](/assets/ml/shap/17.png)

Often, feature importance outputs from the XGBoost Model will contradict each other. 
Depending on the criteria, results will be different.

SHAP values can be used to correctly rank those features.

````python
# this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP Feature Importance
shap.summary_plot(shap_values, X_display, plot_type="bar")
````
![](/assets/ml/shap/18.png)

A variable importance plot lists the most significant variables in descending order. The top variables 
contribute more to the model than the bottom ones and thus have high predictive power.

**Summary Feature Plot**

The SHAP value plot can further show the positive and negative relationships of the predictors with the target variable.

we use a density scatter plot of SHAP values for each feature to identify how much impact each feature has on the model 
output for individuals in the validation dataset. Features are sorted by the sum of the SHAP value magnitudes 
across all samples.

````python
shap.summary_plot(shap_values, X)
````
![](/assets/ml/shap/19.png)

In this example, It's interesting to note that the relationship feature has more total model impact than the 
capital gain feature, but for those samples where capital gain matters it has more impact than age. 

In other words, capital gain effects a few predictions by a large amount, 
while age effects all predictions by a smaller amount.

Through this graph we can get the following information:
- Feature importance: Variables are ranked in descending order.
- Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
- Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.
- Correlations: High/Low variable values with High/Low SHAP impact

## Dependence Plots

The partial dependence plot shows the marginal effect one or two features have on the predicted outcome of a machine learning model.

It tells whether the relationship between the target and a feature is linear, monotonic or more complex.

For individual dependence plots:
````python
feature_name = 'Age'
shap.dependence_plot(feature_name, shap_values, X, interaction_index=feature_name)
````
![](/assets/ml/shap/20.png)

Dependence plots can also be used to assess 2 variable interactions. 
````python
feature_name = 'Education-Num'
interaction_feature = 'Age'
shap.dependence_plot(feature_name, shap_values, X, interaction_index=interaction_feature)
````
![](/assets/ml/shap/21.png)

### Individual Plots

````python
selected_index = 2
shap.force_plot(explainer.expected_value, shap_values[selected_index,:], X_display.iloc[selected_index,:])

selected_index = 2
shap.force_plot(explainer.expected_value, shap_values[selected_index,:], X_display.iloc[selected_index,:],
               link='logit')
````
![](/assets/ml/shap/22.png)

- The output value is the prediction for that observation.
- we can use the logit link function to transform log-odds to probability levels
- The base value: The original paper explains that the base value E(y_hat) is “the value that would be predicted if we 
did not know any features for the current output.” In other words, it is the mean prediction, or mean(yhat). Should be Y_test.mean()
- Red/blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.


### References

https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27

https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d