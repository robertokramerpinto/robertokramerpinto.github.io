# Time Series 

References:
* https://www.machinelearningplus.com/time-series/time-series-analysis-python/
* https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python
* https://www.pluralsight.com/guides/machine-learning-for-time-series-data-in-python
* https://medium.com/analytics-steps/introduction-to-time-series-analysis-time-series-forecasting-machine-learning-methods-models-ecaa76a7b0e3
* https://medium.com/@ODSC/machine-learning-for-time-series-data-e3971d38005b
* https://learning.oreilly.com/library/view/practical-time-series/9781492041641/titlepage01.html
* https://learning.oreilly.com/library/view/hands-on-time-series/9781484259924/html/492113_1_En_1_Chapter.xhtml
* https://www.udemy.com/course/series-temporais-com-python/learn/lecture/16374040#overview

**What is a Time Series ?**

Time series is a sequence of data recorded at regular periods of time intervals.

> Data is presented among time intervals
* These intervals can be days, hours, minutes, seconds, etc.

> There's a natural order between observations, related to time 
    * The time series data needs to be ordered by time

![](/assets/ts/9.png)

**Why to study time series ?**

Time series forecasting plays a major role for several industries. Forecast examples: sales, number of visitors (website),
stock price, etc.

**Panel data**

* Panel data is also a time based dataset
* The difference is that, in addition to time series, it also contains 1 or more related variables that are measured for
the same time periods. 
![](/assets/ts/3.png)

## Time Series Components

![](/assets/ts/4.png)

Any time series may be decomposed into 4 major components:

### Base Level

 
### Trend

- It's the Increasing or Decreasing slope observed in the TS.
- Trend is usually a pattern observed for longer periods of time.

![](/assets/ts/5.png)
![](/assets/ts/8.png) 

### Seasonality
Seasonality is observed when there is a distinct repeated pattern observed between regular intervals (due to seasonal effects).
- Repetition of a cycle for defined time intervals

![](/assets/ts/6.png)
![](/assets/ts/10.png)

#### Cyclic x Seasonal Pattern

Time Series can also present cyclical effects. It happens when the rise and fall pattern observed in the series does not
happen in **fixed/regular calendar-based intervals**. 

Cyclical effect is not the same as a seasonal one. If the patters are not fixed on calendar based frequencies, then it's cyclic. 
Cyclic effects are typically influenced by the business and other socio-economic factors. 

![](/assets/ts/11.png)

### Error
The error is everything that can't be explained by the model. Usually we'll always try to model the time series data using
mathematical models, but we won't be able to precisely explain the data. We can decompose the ts into the base level, 
trend and seasonality under a decomposition model and the part that can't be explained is called the error.

**Note**

It's not mandatory that all time series must have a trend and/or seasonality. 
- A time series may not have a distinct trend but present a seasonality effect.
- The opposite can also be true

### ETS Models - Decomposition

ETS Models refer to Error-Trend-Seasonality Models
* Exponential Smoothing
* Trend Methods Models
* ETS Decomposition

ETS Decomposition model is a specific case for ETS Models. It's used to separate the different components of a TS.

It's a good way to visualize and understand data. 

ETS Decomposition models can also be distinguished between an additive x multiplicative approach.

> Additive Models
* Time series is obtained by the sum of all individual components
* It's more indicated when the trend seems to be more linear with seasonality and trend components seem to be constant
over time (ex: each year we add a contant number to trend, for example)

> Multiplicative Models
* Time series is obtained by a multiplication of individual components
* More appropriate when we are increasing (or decreasing) at a non-linear rate (Ex: each year we double the number observed
in the trend)

**Example**

````python
import pandas as pd
import numpy as np
%matplotlib inline

airline_data = pd.read_csv("UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                          index_col = 'Month',
                          parse_dates=True)

airline_data.head()
airline_data['Thousands of Passengers'].plot()
````
![](/assets/ts/12.png)




# Characteristics and Properties in a TS

## Autocorrelation

A given TS may present autocorrelation: values in t_0 relates to lagged values. 

* It's the mathematical relationship between different time intervals in the ts
* It's measure for different time lags
    * AR(1) --> 1 Lag autocorrelation
    * AR(3) --> 3 lags autocorrelation  
* Varies between -1 and 1 (0 --> no autocorrelation)

It'll indicate if there's a relationship between different points in time for the same time series. 

### How to detect AutoCorrelation ? 

One option is to use the ACF diagram. The ACF graph display the autocorrelation level for the observed ts. 

![](/assets/ts/7.png)

* There's a significance level displayed by a dash line in the graph
* The ACF graph indicates some autocorrelation if values go above this line 




# Time Series Forecasting

