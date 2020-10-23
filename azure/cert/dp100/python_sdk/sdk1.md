# Python SDK 


## Workspace Connection Python SDK

In order to run azure ml pipelines through Python SDK, our initial starting point will be the Azure ML Workspace.

In this scenario, it's possible to create a new workspace or to connect your app to an existing one. Let's see how this 
works.

### Connecting to an existing WS

When connecting to an existing Azure ML Workspace we need first to authenticate our access to the resources. This can 
be done through 2 distinct ways: a manual and an automated process. On the manual process, we pass to the program some
basic information about the WS and the authentication step is done manually through an URL (in case of working with
notebooks for example). Under the automated process, we need to create an application token, give access to that application
inside the existing WS and then connect it automatically. 

In both cases, it's indicated to use a config.json file with WS information. This config.json file can be directly download in the portal
accessing the WS settings. 

Here, we can see an example of such config.json file:
````json
{
    "subscription_id": "974f9871-2375-47c7-bfd5-54e55b74fbdd",
    "resource_group": "cloudgurutraining",
    "workspace_name": "test_1"
}
````

````python
# 01_connect_workspace.py file
from azureml.core import Workspace

def get_ws():
    ws = Workspace.from_config()
    return ws

if __name__=='__main__':
    ws = get_ws()
    print(ws)
````
````
python 01_connect_workspace.py 
````

After running the shell command, a new window will be prompt asking us for manual authentication.

**Automatic Authentication**

When setting up a machine learning workflow as an automated process, it's recommend 
using Service Principal Authentication. This approach decouples the authentication from any specific user login, and allows managed access control.

* Azure Portal >> Azure Active Directory >> Apps Registrations >> Create new application
* Copy Application (client) ID and Tentant (directory) ID 
* Select certificates & secrets (left menu) >> +new client secret >> copy client secret
* Go to IAM from the ML resource and add role to the registered application (add role assignment): grant access to resources for the specific tenant_id.  
* Recommendation is not to hardcode these passwords but instead use environment variables ($env:AZUREML_PASSWORD = "my-password")

````python
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import argparse

class AzureWorkspace():
    def __init__(self):
        self.subscription_id = "974f9871-2375-47c7-bfd5-54e55b74fbdd"
        self.resource_group="cloudgurutraining"
        self.workspace_name="test_1"
        self.my_application_id = "e8f02d4d-9d0b-4abb-93ed-ed365ebee25f"
        self.my_tenant_id = "e2c5fd58-d9bc-4c31-9495-bad58ae11f15"
        self.secret = "_w_pL4RF5h.4FKysfN3.dlqtM~X-2aNtT1"

def get_manual_ws():
    ws = Workspace.from_config()
    return ws

def get_automatic_ws():
    workspace = AzureWorkspace()

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=workspace.my_tenant_id,
        service_principal_id=workspace.my_application_id,
        service_principal_password=workspace.secret)

    ws = Workspace(
        subscription_id=workspace.subscription_id,
        resource_group=workspace.resource_group,
        workspace_name=workspace.workspace_name,
        auth=svc_pr
    )
    return ws


if __name__=='__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument(
        "--ws_auth_type",
        type=int
    )
    args = parser.parse_args()

    type_authentication = args.ws_auth_type
    if type_authentication == 1:
        ws = get_manual_ws()
    elif type_authentication == 2:
        ws = get_automatic_ws()
    else:
        print("Not a valid authentication type")

    print(ws)
````

After running the automated option:

````
python 01_connect_workspace.py --ws_auth_type 2
````

We get the desired output from the Workspace connection:

````
Workspace.create(name='test_1', subscription_id='974f9871-2375-47c7-bfd5-54e55b74fbdd', resource_group='cloudgurutraining')
````

## Running Scripts into AzureML

In this section, let's try to understand the basics of running python scripts in order to execute ML flows in Azure. 

Major objectives in this sections:
* Create python script to be submitted and executed in the Azure ML environment
* View code output in the cloud

In order to run scripts into AZure ML we need to define some initial tasks:
* Create Control Script
* Connect to Workspace
* Create & Attach Experiment
* Define Compute Target
* Run Script

First thing, let's create a source folder inside our project folder, as indicated by Azure ML.

**Control Script**

Next, we need to create a control script to run our desired code. The control script is used to control **how** and **where** our 
main code is executed. So, in this example let's start from the current project structure and hello.py file.

![](/assets/azure/cert/dp100/11.png) 

The control script in this example, looks like this:

````python
# 03_run_hello.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from src.connections import get_automatic_ws


if __name__ == '__main__':

    # Connect to existing Workspace: test_1
    try:
        ws = get_automatic_ws()
    except:
        print("Workspace connection failed...")

    # Create Experiment
    experiment_name = 'experiment_hello'
    experiment = Experiment(workspace=ws, name=experiment_name)

    # Creating Run configuration
    run_config = ScriptRunConfig(source_directory='./src',
                                 script='hello.py',
                                 compute_target='cpu-main-cluster')

    run = experiment.submit(run_config)
    aml_url = run.get_portal_url()
    print(f"Portal URL : {aml_url}")
````
Here, it's important to highlight some aspects of the control script code:

> Workspace()
- The workspace object connects our code to the Azure ML workspace, enabling our code to connect to the necessary
resources

> Experiment()
- the Experiment(...) class provides a simple way to organize multiple runs under a single name
- Later, we can see how our experiments behave and it'll be easier to compare metrics between different runs
- Can be used to try different models and fe steps, for example

> ScriptRunConfig()
- The ScriptRunConfig() class wraps the main function (hello.py) that needs to be executed and passes it to the Workspace
- This is where we configure how we want our script to run in Azure ML environment
- Here we also define the compute target >> The computing resource that will be used to run our code

> run = experiment.submit(run_config)
- submits the main script to be executed
- the submission is called a RUN 

> aml_url = run.get_portal_url()
- URL to monitor the run progress

**Note: Compute Target**

In this scenario, we expect the compute target to be running. 
![](/assets/azure/cert/dp100/12.png) 

**Code Execution**

After executing the control script code in our terminal:
![](/assets/azure/cert/dp100/13.png)

If we go to the our Workspace, we can see our experiment in the Experiments page
![](/assets/azure/cert/dp100/15.png)
![](/assets/azure/cert/dp100/16.png)

We can also explore the logs of the execution in detail: Outputs + logs
![](/assets/azure/cert/dp100/17.png)