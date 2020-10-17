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
 
 ````python
# Let's combine a Grid Search and CV
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

SEED = 42
n_folds = 5

# Model Initiation and Parameters
gbm_params1 = {'learn_rate': [0.05, 0.1],
                'max_depth': [3, 5, 10],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.5, 1.0]}

model = H2OGradientBoostingEstimator(nfolds=n_folds,seed=SEED)

# Grid Initiation and Parameters
search_criteria = {'strategy': "Cartesian"}

cv_grid = H2OGridSearch(model=model,hyper_params = gbm_params1,
                       search_criteria = search_criteria)

#Train grid search
cv_grid.train(x=x, 
           y=y,
           training_frame = df_h1n1,
             seed = SEED)
````
 
 

