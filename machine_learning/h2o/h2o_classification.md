#  H2O

H2O is designed to be a fast, scalable and open-source ML and DL framework. It includes several APIs (Python, R, Scala, Java,...) 
and also a built-in web interface (FLOW) that can be used to complete analytics pipelines as well.

### Algorithms

H2O implements most common ML algorithms:
* GLM (Generalized Linear Models): Linear regression and Logistic Regression
* Naive Bayes
* PCA
* K-means 
* Random Forests
* Gradient Boosting
* and others


## H2O Basic DockerFile

Let's create a docker image so we can work with a controlled and replicable H2O environment. Below we can find a simple example
to deploy h2o into a docker image.

````dockerfile
# Build:
# Start loading initial Image with FROM command. Images are downloaded from Docker HUB
FROM python:3.8-slim

# Install OpenJDK-11
RUN mkdir -p /usr/share/man/man1
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y nano && \
    apt-get install -y htop && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

# Installing h2o
RUN mkdir -p /downloads && cd -P /downloads && \
    curl -O http://h2o-release.s3.amazonaws.com/h2o/rel-zeno/3/h2o-3.30.1.3.zip && \
    apt-get install unzip && \
    unzip h2o-3.30.1.3.zip

# Creating project folder
RUN cd -P / && mkdir ./h2o_tutorial
# Copying files into project folder
WORKDIR ./h2o_tutorial
COPY . .
RUN chmod +x h2o_initial_script.sh

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install jupyter

RUN pip uninstall h2o
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

EXPOSE 8080
EXPOSE 54321

# Start H2O Flow UI
#RUN cd -P /downloads/h2o-3.30.1.3 && java -jar h2o.jar &

````

**Docker Commands**

* Build docker image based on current DockerFile
```shell script
docker build -f Dockerfile -t h2o:latest .
````

* Start running the container (only exposing port 8080 --> jupyter)
````shell script
docker run -p 8080:8080 -ti h2o:latest /bin/bash
````

* If we want to map the 54321 port (h2o default UI) and also to map volumes:
````shell script
docker run -it -p 5000:8080 -p 5001:54321 -v "/Users/kramer roberto/git_repos/h2o_tutorial/notebooks:/h2o_tutorial/notebooks" h2o:latest:latest /bin/bash
````

* To run jupyter inside the container
````shell script
jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
````

* To access running container
````shell script
docker exec -it <cont_name> bash
````

## H2O Implementation

For this example, let's use the flushot dataset (Driven Data Competition). This dataset contains 2 target labels (h1n1_vaccine and seasonal_vaccine.)

````python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Loading raw data
train_features = pd.read_csv("training_set_features.csv")
train_labels = pd.read_csv("training_set_labels.csv")
train = pd.merge(train_features, train_labels, on = ['respondent_id'])

h1n1_train = train.drop(labels=['seasonal_vaccine'],axis=1)
seasonal_train = train.drop(labels=['h1n1_vaccine'],axis=1)
````

Now, let's initiate h2o

````python
import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
````

First we need to parse our data into the h2oFrame object. 

````python
# First we need to load pandas df into H2O Frame
df_h1n1 = h2o.H2OFrame(h1n1_train)

print(df_h1n1.types)

# Identify predictors and response
x = df_h1n1.columns
y = "h1n1_vaccine"
x.remove(y)

# For binary classification, response should be a factor
df_h1n1[y] = df_h1n1[y].asfactor()
````
.types method will output all detect types. Enum indicates categorical features. 
 
We also need to define x and y (which are our features and target). The target variable for binary classification problems 
also need to be indicated as a factor type (like a categorical one).
 
### Gradient Boosting Model 
 
Let's now explore a GridSearch CV for the GBM. 

The key here is to start tuning some key parameters first (i.e., those that we expect to have the biggest impact on the results). 
From experience with gradient boosted trees across many datasets, we can state the following "rules":

* Build as many trees (ntrees) as it takes until the validation set error starts increasing.
* A lower learning rate (learn_rate) is generally better, but will require more trees. 
Using learn_rate=0.02and learn_rate_annealing=0.995 (reduction of learning rate with each additional tree) can help speed up 
convergence without sacrificing accuracy too much, and is great to hyper-parameter searches. For faster scans, use values of 0.05 and 0.99 instead.
* The optimum maximum allowed depth for the trees (max_depth) is data dependent, deeper trees take longer to train, especially at depths greater than 10.
* Row and column sampling (sample_rate and col_sample_rate) can improve generalization and lead to lower validation and test set errors. 
Good general values for large datasets are around 0.7 to 0.8 (sampling 70-80 percent of the data) for both parameters. 
* Column sampling per tree (col_sample_rate_per_tree) can also be tuned. Note that it is multiplicative with col_sample_rate, so setting both parameters to 0.8 results in 64% of columns being 
considered at any given node to split.
* For highly imbalanced classification datasets (e.g., fewer buyers than non-buyers), stratified row sampling based on response class membership can help improve predictive accuracy. 
It is configured with sample_rate_per_class (array of ratios, one per response class in lexicographic order).

Most other options only have a small impact on the model performance, but are worth tuning with a Random hyper-parameter search nonetheless, if highest performance is critical.

First we want to know what value of max_depth to use because it has a big impact on the model training time and optimal values depend strongly on the dataset. 
We'll do a quick Cartesian grid search to get a rough idea of good candidate max_depth values. 
Each model in the grid search will use early stopping to tune the number of trees using the validation set AUC, as before. 
    We'll use learning rate annealing to speed up convergence without sacrificing too much accuracy. 
    
````python
# Let's combine a Grid Search and CV
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

SEED = 42
n_folds = 5

# Model Initiation and Parameters
gbm_params1 = {'learn_rate': [0.05, 0.1],
                'max_depth': list(range(1,15,2)),
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.7,0.9,1.0]}

model = H2OGradientBoostingEstimator(nfolds=n_folds,seed=SEED,
                                    ntrees=1000,
                                    stopping_rounds = 5,
                                    stopping_metric = "AUC",
                                    stopping_tolerance = 1e-4)

# Grid Initiation and Parameters
search_criteria = {'strategy': "Cartesian"}

cv_grid = H2OGridSearch(model=model,hyper_params = gbm_params1,
                       search_criteria = search_criteria)

#Train grid search
cv_grid.train(x=x, 
           y=y,
           training_frame = df_h1n1,
             seed = SEED)

# Get the grid results, sorted by validation AUC
grid_results = cv_grid.get_grid(sort_by='auc', decreasing=True)
grid_results[0]

````
In our example, this was the best model (results)

![](/assets/ml/h2o/2.png)
![](/assets/ml/h2o/3.png)
 
We can also run a Random Search Grid based on our previous estimated parameters values. 

````python
# create hyperameter and search criteria lists (ranges are inclusive..exclusive))
hyper_params_tune = {'max_depth' : [4,5,6],
                'sample_rate': [0.7,0.8,0.9,1],
                'col_sample_rate' : [0.7,0.8,0.9,1],
                'nbins': [2**x for x in [4,5,7]],
                'nbins_cats': [2**x for x in [4,5,7]]}

search_criteria_tune = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 3600,  ## limit the runtime to 60 minutes
                   'max_models': 100,  ## build no more than 100 models
                   'seed' : SEED,
                   'stopping_rounds' : 5,
                   'stopping_metric' : "AUC",
                   'stopping_tolerance': 1e-3
                   }

gbm_final_grid = H2OGradientBoostingEstimator(distribution='bernoulli',
                    nfolds=n_folds,
                    ## more trees is better if the learning rate is small enough 
                    ## here, use "more than enough" trees - we have early stopping
                    ntrees=10000,
                    ## smaller learning rate is better
                    ## since we have learning_rate_annealing, we can afford to start with a 
                    #bigger learning rate
                    learn_rate=0.05,
                    ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                    ## (use 1.00 to disable, but then lower the learning_rate)
                    learn_rate_annealing = 0.99,
                    ## score every 10 trees to make early stopping reproducible 
                    #(it depends on the scoring interval)
                    score_tree_interval = 10,
                    ## fix a random number generator seed for reproducibility
                    seed = SEED,
                    ## early stopping once the validation AUC doesn't improve by at least 0.01% for 
                    #5 consecutive scoring events
                    stopping_rounds = 5,
                    stopping_metric = "AUC",
                    stopping_tolerance = 1e-4)

#Build grid search with previously made GBM and hyper parameters
final_grid = H2OGridSearch(gbm_final_grid, hyper_params = hyper_params_tune,
                                    grid_id = 'final_grid',
                                    search_criteria = search_criteria_tune)
#Train grid search
final_grid.train(x=x, 
                 y=y,
                 training_frame = df_h1n1,
                 seed = SEED,
                 max_runtime_secs = 3600)
````

