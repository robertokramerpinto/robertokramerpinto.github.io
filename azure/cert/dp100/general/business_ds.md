# Data Science Lifecycle

![](/assets/azure/cert/dp100/1.png)

There are 5 major stages in the Data Science Lifecycle: 

> Business Understanding
- Problem to solve
- Identify data sources
- Determine accuracy

> Data Acquisition
- Data Ingestion (Azure Blob, SQL DB, Hive Tables)
- Data Ingestion Pipeline (Using tools like DataFactory, Databricks, Airflow) 
- Data Wrangling
- Data Exploration

> Modeling
- Feature Engineering
- Model Training 
- Model Evaluation

> Deployment 
- Expose model with open API interface
- Dump data into DBs

> Customer Acceptance
- Presenting ML results to end user 
- System validation
- Project hand-off --> move it into production

### Business Understanding

#### Defining Objectives

Defining objectives: Business understanding & SMART metrics

Here the main goal is to define business problems/solutions using advanced analytics. We here need to start identifying
what data we have/need in order to solve the problem. 

To define the problem properly, we need to start asking questions. We can do that through the SMART Metrics approach.

**SMART Goal**

SMART relates to the process of creating Specific, Measurable, Achievable, Relevant and Time-Related objectives.

- Specific: 5 Ws (Who, what, why, where, which)
- Measurable: Be able to track progress
- Achievable: Make your goals ambitious, but achievable
- Relevant: Ensure that your goal is relevant to the company's vision & creates value
- Time-Related: Use deadlines to track progress

**Identify Data Sources**
- Select only data that's relevant to the objectives (scope) of the project
- Is the data an accurate measure of the model target?

**Artifacts**

We can also use artifacts to help the project development at this initial stage. Artifacts are documents outputs at each 
step of the project. 
 
* Charter document
    * Early level documentation of the project (risks, costs, major goals, milestones)
* Data sources
    * Location and data to be used in the problem
    * Data dictionaries (describe data in data sources, schemas, entity relation documents) 