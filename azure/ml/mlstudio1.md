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





