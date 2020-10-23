# Running Scripts into AzureML

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