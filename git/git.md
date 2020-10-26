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

### SHA hashes


 
