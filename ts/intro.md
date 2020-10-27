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

![](/assets/ts/2.png)

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
 

### Seasonality
Seasonality is observed when there is a distinct repeated pattern observed between regular intervals (due to seasonal effects)

#### Cyclic x Seasonal Pattern
Time Series can also present cyclical effects. It happens when the rise and fall pattern observed in the series does not
happen in **fixed calendar-based intervals**. 

Cyclical effect is not the same as a seasonal one. If the patters are not fixed on calendar based frequencies, then it's cyclic. 
Cyclic effects are typically influenced by the business and other socio-economic factors. 

### Error


**Note**

It's not mandatory that all time series must have a trend and/or seasonality. 
- A time series may not have a distinct trend but present a seasonality effect.
- The opposite can also be true

