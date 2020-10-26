# Installation

**Linux**

````
apt-get update
apt-get upgrade -y
apt-get install -y git
````

```` git --version```` displays the version of git installed in your system

```` which git```` displays the current bin directory where git is installed

```` man git```` displays main git manual page

# Git Local Repo

````git init <path_to_project_folder>```` Initializes a git local git repository 
* If the path is empty a new folder will be created with git initial files (.git/)
* If we execute this command to an existing folders, we'll be adding the git repository files to it

````ls -lh .git/```` list all files within .git/ repo folder



