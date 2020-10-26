# Source Control with GIT

# Git Installation 

**Linux (Debian/Ubuntu) System**

For a git basic installation in a Linux Server we can use the following commands:

````
apt-get update
apt-get upgrade -y
apt-get install -y git
````

**RedHat based system**

````
yum install git # installs git on local system from distribution's repositories
````

After installed we can check its current version by executing:

````
git --version
````

# Git File System

Whenever we launch a git repository, git will create a hidden folder `````.git````` inside our working directory.

> .git
* Contains the configuration information about repository
* Contains all the tracking information about all files within the project folder
* It contains a DB file that git uses to keep tabs on all files added to the project. 

Initially when we add a new file to our project folder, git is unaware of it. Git only starts tracing the file
when we ````add```` it to gits stage area.

![](/assets/git/1.png)

## Staging Area

The staging area is just a virtual location that git uses to track changes to files. 
Git will only be able to detect changes in files if they're added to the staging area first.

Git is not moving files physically, it's just recording the **state** of the file contents by
using SHA one hashes. 

![](/assets/git/2.png)

## Branches

Within our repository, we'll have lines of development known as **branches**.

When we commit a change to a project (persist the detect files modifications into git's database), we're attaching the change
state of the file to a given branch in our repository. The current file state leaves the staging area and is recorded to a given branch.

![](/assets/git/3.png)

The **master branch** is typically the default one. 

We can also have multiple branches in our repository. Usually the master branch is used for the tested and latest deployed version of our project and we use 
other branches to make changes, updates, tests etc. This process is called branching strategy and can have multiple designs. 

![](/assets/git/4.png)

Example: In order to change a color layout in our website, we can create a feature branch to perform and check the new concept
and if it's approved, we can merge the feature branch into the master branch so the modifications can be captured by the master 
branch and available in production. 

## Git Repo List of files

This is typical list of files inside the .git folder

![](/assets/git/5.png)

> COMMIT_EDITMSG
* text file containing the commit messages
* the order in the file corresponds to the commit order (1st line --> 1st commmit)

> HEAD
* text file with reference to latest file

> config
* Contains configuration information about the repository
* developers email and username

> index
* keeps track of files in the staging area

> hooks directory
* used to store some particular scripts to automate processes in the project

> objects 
* Folder that contains hashed versions of content files that have been committed
* It's our git DB

# Local Repository

We can create locally our repository through git commands or we can directly clone a remote
repository in our local system. 

## Creating a local repo










 
