# Azure Machine Learning Studio

Azure Machine Learning is a cloud-environment designed to to train, deploy, automate, manage, and track ML models.
Azure Machine Learning can be used for any kind of machine learning, from classical ml to deep learning, supervised, and
unsupervised learning. Whether you prefer to write Python or R code with the SDK or work with no-code/low-code options in 
the studio, you can build, train, and track machine learning and deep-learning models in an Azure Machine Learning Workspace.


![](/assets/azure/ml/2.png)


Azure ML Studio is a web interface application that can be used to facilitate E2E Machine learning solutions.

Among its features, we can highlight:
* Easy to understand/visual ML tool
* User can interact with the platform with a no/low code requirement approach. It's a drag-drop solution.  
![](/assets/azure/ml/3.png)
* ML Studio also provides great integration with Azure Data Platform tools (like Data Factory, Stream Analytics and Data Lake).
![](/assets/azure/ml/1.png)
* When ready we can publish our ML output as a web-service

## Creating Azure ML Studio Workspace

First we need to go to resources and select the Machine Learning Resources. After that we need to enter the required 
information in order to validate and create the resource. Once the ML resource is ready we can enter access it and 
select the ML Studio button. Next, we need to select account and resource group in order to validate and create our 
ML Studio Environment. 

![](/assets/azure/ml/4.png)

Inside the ML Studio environment we have several available features to explore when building our ML pipeline.
Some of these features are:
> Notebooks

> Experiments

> Ml Designer
- Drag-Drop tool designed to build E2E ML applications

> Models

> Endpoints


## ML Basic Workflow

The ML Studio Designer tool is used to create E2E ML pipelines through a very user-friendly interface (drag-drop tools).
User can also check several Tutorials that might help when designing your pipeline. 

The Designer offers the user a rich list of assets to be used in the ML solution. 
![](/assets/azure/ml/5.png)

A basic ML workflow can be structured as follows:
![](/assets/azure/ml/6.png)

### Data Ingestion

The ML Designer provide us several data ingestion tools
* Dataset
* Sample Datasets (testing)
* Data Input and Output artifacts
    * Here we can connect our ML pipe with Azure data solutions

In our example, let's select the Adult Census Income Binary Classification sample dataset provided by Azure. 
![](/assets/azure/ml/7.png)

It's possible to explore the dataset output by right-clicking the data icon. Then inside the data explorer, we can check
the initial rows and see how real data looks like. It's also possible to select a specific column to get more detailed 
information about it: missing values, unique values, distribution metrics (mean, median, max, min), etc.

![](/assets/azure/ml/8.png)

### Data Processing

Here is the step where we'll be processing and modifying our data. 

Let's look at one simple example, where we load our initial data, create the target variable and drop/select cols.
![](/assets/azure/ml/9.png)

In the first step we're calling a SQL transformation where we create a "target" column based on the initial dataset.
The second step is the data selection one, where we're dropping irrelevant cols and selecting the new target variable
(excluding the old one).
![](/assets/azure/ml/10.png)

After executing the initial steps we can always check the output dataset. 

Next steps consist in filling missing values for both numerical and categorical columns. After that, we split our 
dataset for training and validation (using stratified distribution based on the target variable).

Important to say that the data split step outputs 2 datasets, the left one is the one which you define the split rate
(70% in our example, the one we'll be using as the training set as well).

![](/assets/azure/ml/11.png)

### Modeling

For the modeling phase, let's select the Train model function. Here we have 2 inputs: left --> algorithms, right -->
training set. The model we'll be using in this example is the binary logistic regression algorithm with l2 penalization (LASSO).

After training our model using the selected training set from split data step, we can score the predictions based on
the test set and evaluate our test results. 

![](/assets/azure/ml/12.png)

After the training and scoring process is completed, we can check the evaluation output.
![](/assets/azure/ml/13.png)



 




#### Data Transformation Steps

> Select Columns in Dataset
- As the name says, user can select a subset of columns from the input dataset

>Edit Metadata
- This step allow us to rename column names and modify data types





