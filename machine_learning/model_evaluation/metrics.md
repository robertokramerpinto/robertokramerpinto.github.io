# Metrics

## Regression

Regression models are used when predicting **continuous** variables.

### MAE (Mean Absolute Error)
![](/assets/ml/theory/3.png)

It's the average absolute difference between predictions and observed (truth values). Idea is to compute the absolute error
for each point and then average them to get a general estimation of the error. 

* Simple and intuitive metric
* Absolute values are used to treat both positive and negative errors equally
    * avoid errors to compensate each other
* Treats all points equally
* Less sensitive to outliers (more robust to outliers)
    * If your data presents outliers this metric can be more appropiate
    * Avoid having large errors impacting heavily the overall metric
* Output presents the same scale as the target variable

````python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_predictions)
````

### MSE (Mean Squared Error)
![](/assets/ml/theory/4.png)

It's the average square difference between predictions and observed (truth) values.

* Most widely used regression error metric 
* Sensitive to outliers
    * It allows larger errors to have a larger impact on the model
    * Outliers errors will contribute more to the general model error

````python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_predictions)
````

