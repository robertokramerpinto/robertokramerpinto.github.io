# Time Series Forecasting

We can perform time series forecasting based on several families of models. In general terms, we can 
divide those models in between 3 major groups: statistical models, machine learning models and deep learning models. 

All forecasting models will present "errors"/deviations from the real observed values and perhaps some types of models
will be more suitable for certain types of problems and data. Our modeling stage will be as good as the explanatory 
variables and models we dispose. 

> Main Objective
- In theory, all forecast models will present errors. Main goal is to minimize such error.

## Problem definition and Business insights

Whenever approaching a real world problem we should always ask ourselves: If I knew ___, I would improve this ___,
because of ____.

And also, whenever we're defining a forecasting problem we need to go through several questions like:
* What do we want to forecast ? 
* How are we currently forecasting ?
* What are the real world constraints for your forecasts ?
* Are there any Series we don't want to Under/Over forecast ?
* How far do we need to forecast ?
* Are there relevant features available that can be used in a model ?
* Will those features also be available when we're forecasting new data ?
* Is there a minimum forecast unit due to physical/business limitations?
    * Item, case, pallet, etc.
* Do we have many new Series ?
* Are there any specific days we're concerned about?
    * Holidays, Black Friday, etc
* What time period are we going to evaluate the model on?
* Are the items we are forecasting highly seasonal ? 
* Is there any time constraint for the forecasted items?
* What is the metric/accuracy we need to exceed ?
* How granular do your forecasts need to be?
* Recent Changes
    * Competitors entering/exiting market
    * Mergers/Acquisitions 
    * Have line of business changed?
    * Has the ecommerce site changed?
    * Have multiple stores closed/opened?

### Granularity

In terms of granularity (time interval) we can distinguish some trade-offs between more and less granularity

> Less granularity (weekly, quarterly, monthly, yearly)
* More stable target
* Less data to train models
* Dynamics are often damped
* Tend to be worse ML problems

> More granularity (seconds, minutes, hours, days)
* Less stable target
* More data to train models
* Higher likelihood models will capture interesting dynamics
* Potentially more missing data

### Deliver Impact 

Always critical to focus on high-value use cases

![](/assets/ts/14.png)

## Modeling

Whenever we model time series, it's also important to ask relevant questions about this part:
* Single-Series or Multi-Series?
    * One model for each item
    * One model for several items (items desc goes as feature)
* How far back do we create lags?
* Which optimization metric to use?
* Do we have all features to be used in advance ?
* Do we have a calendar file?


