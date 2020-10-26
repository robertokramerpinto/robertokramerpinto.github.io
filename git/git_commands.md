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

## Repository configuration

````git config```` Base command to configure several elements of git env

Whenever we start a git environment we need to define at least 2 configuration parameters: username and email.
This information is relevant only to track who made changes to the files. 

````git config --global user.name "<username>"  ```` 

````git config --global user.email "<useremail>"  ````
* Base command to configure several elements of git env
* global flag is used to pass same user and email to all projects

If instead, we just want to change configuration for a particular project --> do not use --global flag.

````git config user.name "<username>"  ```` Define username for one particular project (not global)
* This command needs to be executed inside the project's working directory
* In this situation, git will add an extra line for the property in the config file and will use only that latest values of all properties


````git config --list```` Displays configuration file information

There are also configuration properties we can change:

````git config --global core.editor "usr/bin/vim```` Set default text editor
* can use nano, etc just need to include all path to binary editor

Example:

````shell script
[cloud_user@ip-10-0-0-221 ~]$ git config --global user.name "cloud_user"
[cloud_user@ip-10-0-0-221 ~]$ git config --global user.email "cloud_user@mylabserver.com"
[cloud_user@ip-10-0-0-221 ~]$ git config --global core.editor "/usr/bin/vi"
[cloud_user@ip-10-0-0-221 ~]$ git config --list
user.name=cloud_user
user.email=cloud_user@mylabserver.com
core.editor=/usr/bin/vi
[cloud_user@ip-10-0-0-221 ~]$ 
````

**Configuration file**

All changes will also appear in our root configuration file ````~/.gitconfig````

````cat ~/.gitconfig```` Outputs saved configuration properties

## Handling Files into our Project 

**Adding Files** 

In order to make git start tracking our files (which live inside our working directory) we need to first add them to the staging area.

````git add <file1> <file2> ... ```` Add files to the index file so they can be tracked in the staging area

**Status**

````git status```` Check current branch and files that are in the staging area (not committed yet)

````git status -s```` Status output in a short version
* A: indicates a file is added to staging area
* ??: indicates a file is not tracked
* M: indicates a file is modified
* D: deleted file

````git status -v```` Status output with more verbose output, including changes to files

**Removing Files**

````git rm <file1> <file2>```` Removes files from a project

### Committing Files

* A commit command is used to persist files versions in git db.
* A commit always need to have a message
* A commit will only persist files that exist in the staging area

````git commit```` Opens the text editor to receive a commit message and then commits to git DB

````git commit -m "<commit message>```` Performs a commit with the specified message

````git commit -a -m "<commit message>```` Commit a modified file in the staging area

### Ignoring Files

````.gitignore```` File used to manage files we don't want to track 
* Ignore files based on patterns

Inside the .gitignore file we can have patterns for data or notebook files. We can also use this file to avoid tracking
passwords and sensitive information. Example:

````ignore
# IDE settings
.vscode
.idea
.DS_Store

## Local files
# Credentials
# SSH keys (.pem files)
# Secret config files

# Data
*.csv
*.h5
````













