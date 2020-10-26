# Azure Machine Learning 

Azure machine learning is an fully equipped cloud environment designed to deliver end-to-end ML solutions.  In summary,
it's a cloud-based environment you can use to train, deploy, automate, manage, and track ML models.

It integrates several Azure services, from data, modeling, DevOps and other tools, facilitating a robust implementation 
of complex analytical solutions. 

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

In general terms, machine learning workflows can be processed using Azure Machine Learning resources from distinct ways:
- Azure ML Studio 
- Python SDK (Software Development Kit)
- Azure ML CLI (Commmand Line Interface)

All ways will provide the user functions to run ml workflows using Azure ML Resources & Tools. 


# Azure ML Workspace

The Azure ML workspace is the top-level resource for Azure Machine learning. 

![](/assets/azure/cert/dp100/10.png)

It's a centralized environment used to:
- Manage all resources used to build,train and deploy ml models
- Store assets created during the ML workflow
- Allows users to create and work with artifacts 
- Provides a history of training runs (Experiments)

Each workspace is tied to an Azure subscription and resource group, and has an associated SKU. Ideally each project should
have its own Azure ML Workspace.  

The workspace keeps a history of all training runs, including logs, metrics, output, and a snapshot of your scripts. 
You use this information to determine which training run produces the best model.

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


## Creating an Azure Machine Learning Workspace
### Create an azure ml workspace

#### Using the Azure Portal
To create a ML workspace through the Portal, we can search for the machine learning resource, as indicated below:

![](/assets/azure/cert/dp100/23.png)

It's necessary to assign a resource group, a name for the workspace, select the region and the user can also configure
advanced network connection properties to allow public/private connections as well. 
![](/assets/azure/cert/dp100/24.png)

After validation and creation steps, we will be able to access the ML workspace environment.
![](/assets/azure/cert/dp100/25.png)


### Configure workspace settings
### Manage a workspace by using Azure ML Studio
