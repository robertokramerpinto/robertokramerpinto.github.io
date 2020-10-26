# Model training

Let's see an example of how to train a binary classification model in Azure. 

Main objectives of this sections are:
- create and execute a training script
- create a control script
- Use Logging
- Understand and use Azure ML Classes (Environment, Run, Metrics)
- Submit, run and review training script outputs

## Creating a simple training script (sklearn models)

In this example, let's see how to use Azure ML resources to train a sklearn model. 

Steps:

> Create env
- Use conda env and yml file

> Create aux modules and train main script
- Use sklearn and auxiliary modules

> Create Control Script
- Connect to Workspace
- Create & Attach Experiment
- Define Compute Target
- Run Script
 

### 1. Create env
For this example, let's use the conda env (easy integration with azure ml).

Idea is to create a local env that can be tested before sending code to the cluster to be executed remotely. We need 
to create a env yml file so our cloud compute instance can also replicate the same environment. 

Let's start by creating the env yml file. This file goes under the .azureml folder: 
````yaml
name: ml-env
dependencies:
    - python=3.6.2
    - pandas
    - scikit-learn
    - pip
    - pip:
        - azureml-sdk
````

After that we can create our local virtual conda environment:
````python
conda env create --force -f .azureml/ml-env.yml     # create conda environment
conda activate ml-env                               # activate conda environment

#python src/train.py                                # train model
#conda deactivate                                   # deactivate conda environment
````

### 2. Train File

First let's create our ````model.py```` script that will serve as reference for the ````training_bin_model.py```` script.

````python
# train.py
from model import get_logistic_reg_model

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from azureml.core import Run

# Get Azure Machine Learning run from the current context
run = Run.get_context()

def get_data():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


# Loading Data
X_train, X_test, y_train, y_test = get_data()
print("Breast Cancer Data Loaded")

# Initiating Model
model_params = {"random_state":42, "C":1.0, "max_iter":5000}
model = get_logistic_reg_model(params=model_params)
print("Model loaded")

# Training Step
model.fit(X_train, y_train)
print("Model fitted to training set")

# Predictions
print("Making Predictions to train and test sets")
y_train_scores = model.predict_proba(X_train)[:,1]
y_test_scores = model.predict_proba(X_test)[:, 1]

# Metrics
roc_auc_train = roc_auc_score(y_train, y_train_scores)
run.log('roc_auc_train', roc_auc_train)
print(f"ROC AUC Train: {round(roc_auc_train,4)}")
roc_auc_test = roc_auc_score(y_test, y_test_scores)
run.log('roc_auc_test', roc_auc_test)
print(f"ROC AUC Test: {round(roc_auc_test, 4)}")
````

````python
# model.py
from sklearn.linear_model import LogisticRegression

def get_logistic_reg_model(params):
    model = LogisticRegression(**params)
    return model
````

After creating the train script we can locally execute it:
![](/assets/azure/cert/dp100/19.png)

### 3. Control Script
Let's create our control script: ````04_run_train_bin_class.py````

````python
# 04_run_train_bin_class.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from src.connections import get_automatic_ws


if __name__ == '__main__':

    # Connect to existing Workspace: test_1
    try:
        ws = get_automatic_ws()
    except:
        print("Workspace connection failed...")

    # Create Experiment
    experiment_name = 'breast_cancer_modeling'
    experiment = Experiment(workspace=ws, name=experiment_name)

    # Creating Run configuration
    run_config = ScriptRunConfig(source_directory='./src',
                                 script='train.py',
                                 compute_target='cpu-main-instance')

    # Define Environment
    env = Environment.from_conda_specification(name='ml-env', file_path='.azureml/ml-env.yml')
    run_config.run_config.environment = env
    
    # Submit Files
    run = experiment.submit(run_config)
    aml_url = run.get_portal_url()
    print(f"Portal URL : {aml_url}")

````

Next, let's execute the control script:

````python 04_run_train_bin_class.py````

![](/assets/azure/cert/dp100/20.png)

After executed, we can check our logging metrics:
![](/assets/azure/cert/dp100/21.png)

#### Logging
In ````train.py```` we can access the run object from within the training script itself by using the ````Run.get_context()````
method and use it to log metrics:

`````python
# in train.py
run = Run.get_context()
...
run.log('loss', loss)
`````
Metrics in Azure Machine Learning are:
- Organized by experiment and run, so it's easy to keep track of and compare metrics.
- Equipped with a UI so you can visualize training performance in the studio.
- Designed to scale, so you keep these benefits even as you run hundreds of experiments.

**Folder Structure**

This is the output of our current folder structure:
![](/assets/azure/cert/dp100/22.png)

## Registering and Deploying Model

Now, let's see how we can improve a bit our ML solution. In this section, let's add some layers to the process:
- Read input data from Blob storage
- Save and Register model
- Deploy model









