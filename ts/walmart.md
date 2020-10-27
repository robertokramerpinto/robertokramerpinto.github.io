# Case: Walmart Store Sales Forecasting

https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

## Data Description

> Objective
* predict the department-wide sales for each store

> Features
* Historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments
* Walmart runs several promotional markdown events throughout the year. 
* These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. 
* The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. 

Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.

> Files
* stores.csv
    * This file contains anonymized information about the 45 stores, indicating the type and size of store
* train.csv
    * This is the historical training data, which covers to 2010-02-05 to 2012-11-01. 
    * Within this file you will find the following fields:
* test.csv
    * Output file expecting predictions
* features.csv
    * Additional data related to the store, department and region

## Exploratory Data Analysis

````python
#!kaggle competitions download -c walmart-recruiting-store-sales-forecasting
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline 

# Loading Initial Data
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")
````
![](/assets/ts/15.png)

