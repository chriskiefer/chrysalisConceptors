# Chrysalis and conceptors

This repository contains the code developed during a residency at Sussex University, developing new algorithms to be used in the piece [Chrysalis](https://marijebaalman.eu/chrysalis).

# links

* http://minds.jacobs-university.de/conceptors
* http://minds.jacobs-university.de/ESNresearch

# Setup

We use two programming languages, SuperCollider and Python3.

# Setup for python

Python3 is used within a virtual environment, to keep it contained. We use jupyter-notebooks for interactive execution of the code.

Steps for the setup are:

* On Linux (Debian/Ubuntu) install the python3-dev files (needed for pyliblo below) and virtualenv

```
sudo apt-get install python3-dev virtualenv
```

* make a virtualenv - just making it (I've called mine `env`)

```
virtualenv -p python3 <dirname>
```

* to enter the virtual environment

```
source <dirname>/bin/activate
```

* packages (each of the packages can be installed with `pip install <packagename>` while in the virtual environment)

```
    numpy
    scipy
    matplotlib
    jupyter
    Cython
    pyliblo
    dill
```

* Start the jupyter notebook (browserbased interactive editor)

```
jupyter-notebook
```


* to deactivate the virtual environment and leave it

```
deactivate
```
