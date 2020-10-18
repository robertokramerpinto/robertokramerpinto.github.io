# AutoML 

![](/assets/ml/h2o/4.png)

H2O's AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of 
many models within a user-specified time-limit. It also allows stacked models, a combination of single models
that will generally be placed as the top performance models. 

AutoML uses the same data-related arguments, x, y, training_frame, validation_frame, as the other H2O algorithms. Most 
of the time, all you’ll need to do is specify the data arguments. You can then configure values for max_runtime_secs 
and/or max_models to set explicit time or number-of-model limits on your run.

So, as required parameters for tuning AutoML we have:

> Required Data Parameters
- y (string of target col) and training_frame (training set)

> Required Stopping Parameters
- max_runtime_secs (max time h2o will take to run models)
- max_models (max number of models AutoML will build)





