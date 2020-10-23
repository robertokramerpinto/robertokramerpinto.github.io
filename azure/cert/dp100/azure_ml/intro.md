# Azure Machine Learning 

Azure machine learning is an fully equipped cloud environment designed to deliver end-to-end ML solutions.  In summary,
it's a cloud-based environment you can use to train, deploy, automate, manage, and track ML models.

It integrates several Azure services, from data, modeling, DevOps and other tools, facilitating a robust implementation 
of complex analytical solutions. 

![](/assets/azure/cert/dp100/2.png)

**Why to use Azure Machine Learning?**
- Simplify ML building models with AutomatedML
- Easy scaling model training in a cloud environment
- Use Python SDK and any Python open source frameworks & tools
- Manage workflows with DevOps for machine learning
- Simple deployment with cloud hosted endpoints

This is also a general picture of the end-to-end Ml cycle integrating several Azure ML tools:
![](/assets/azure/cert/dp100/9.png)

User can interact with Azure ML resources in several ways. For example:
![](/assets/azure/cert/dp100/8.png)



**Azure ML Workspace**

- The Azure ML Workspace is the top level resource for Azure ML solutions.
- Centralized Environment to use ML Tools and objects

A workspace is used to experiment, train, and deploy machine learning models. It's a centralized place to work with all 
the artifacts you create when you use Azure Machine Learning. 
The workspace keeps a history of all training runs, including logs, metrics, output, and a snapshot of your scripts. 
You use this information to determine which training run produces the best model.

- Each workspace is tied to an Azure subscription and resource group, and has an associated SKU.

### Major Components of the Azure Machine Learning Workspace

![](/assets/azure/cert/dp100/3.png)

> Compute Instances
- Cloud resources with Python environment
- User can use compute instances to develop and run ML pipelines independently 

> User Roles
- Share a workspace with other users, teams or projects

> Compute Targets
- Used to run experiments
- From a compute instance we can attache ML pipelines to compute targets so our jobs can be executed

> Experiments
- Train runs to build models

> Pipelines
- Reusable workflows for training and retraining models

> Datasets
- Aid with data management

> Registered Model 
- Created after we are ready to deploy our models

> Notebooks 
- We can launch notebooks to run ML pipelines

> Automated ML
- AutoML framework to develop automated ml pipelines

> Deployment Endpoint
- Final step, a combination of the registered model and scoring script

## Compute

![](/assets/azure/cert/dp100/5.png)

Compute resources will provide us the computation power we need to execute our ML workflow. They are divided into 2 major
groups: compute instances and clusters. 

### Compute Instance 

We can choose from a selection of CPU or GPU instances preconfigured with popular tools such as JupyterLab, Jupyter, 
and RStudio, ML packages, deep learning frameworks, and GPU drivers to process our ML workflows.

![](/assets/azure/cert/dp100/6.png)

When creating a compute instance we can choose CPU/GPU computation, memory and disk sizes, as well as advanced
network configuration. 