# setup for the workshop

* Follow the main setup guideline in the general [README.md](../README.md) of this repository for the python virtual environment and jupyter-notebook

* For SuperCollider, follow the instructions at the top of [datainput.scd]() to install the needed Quark.


# files we use during the workshop

In the workshop we will switch between SuperCollider and python (in jupyter-notebooks).

# gesture recording

`datainput.scd`

In this file we will record gestures, for this we will receive data from one of the accelerometers, do some preprocessing in SuperCollider and save the gestures to disk.

# training a network

`gestures_detection.ipynb`

Then we switch to Python to train a network to these gestures in the file `gestures_detection.ipynb`

At the end of this we save the network for later use.

# detect a gesture in realtime

`realtime_recognition.scd`
`detectgesture_realtime.ipynb`
`visualise_detection.scd`

In SuperCollider we now send the data to python. In python we load the network and listen for the incoming data and do a realtime estimation of which gesture it is. The output is received back in SuperCollider, where we do a realtime plot of the data. Then we also create a simple visualisation of the detection which give a more clear indication of how this works.

