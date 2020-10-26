# Docker Commands

## Images

* List available images (locally)
    * `````docker image ls`````

* Build Image
    * From Dockerfile
        * ````docker build -f Dockerfile -t flushot:train .````
    * Airflow Spark image example 
        * ````docker build --rm --build-arg AIRFLOW_DEPS="gcp,statsd,sentry" --build-arg PYTHON_DEPS="pyspark==2.4.5" --build-arg AIRFLOW_VERSION=1.10.10 -t syn_merch .````

## Containers

The run command creates a new containe based on a docker image.

* Run container 
    * ````docker run -d -p 8080:8080 -p 5050:5050 syn_merch webserver````
        * -p : port mapping
        * Command executed after the image is created using a webserver entrypoint (airflow example)

    * ````docker run -it -p 8080:8080 -v "/Users/kramer roberto/git_repos/flushot_learning/notebooks:/home/flushot_learning/notebooks" e5ff2aedc935 /bin/bash````
        * Run Container with volume and bash interaction

* List existing containers
    * ````docker container ls````
        * active containers
    * ````docker container ls -a ````
        * list all containers
* Access existing running container 
    * ````docker exec -it <cont_name> /bin/bash````
    * ````docker exec -it --user root <cont_name> /bin/bash````

* Stop and Start Container
    *````docker stop <container name>````
    * ````docker start <container name>````

* Delete all containers
    * ````docker rm $(docker ps -a -q)````