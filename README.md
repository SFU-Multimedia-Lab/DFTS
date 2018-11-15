# DFTS: Deep Feature Transmission Simulator
---------------------------------

DFTS is a simulator intended for studying deep feature transmission over unreliable channels. If you use this simulator in your work, please cite the following paper: 

H. Unnibhavi, H. Choi, S. R. Alvar, and I. V. Bajić, "DFTS: Deep Feature Transmission Simulator," demo paper at *IEEE MMSP'18*, Vancouver, BC, Aug. 2018. [[link](https://www.researchgate.net/publication/327477545_DFTS_Deep_Feature_Transmission_Simulator){:target="_blank"}]

## Contents

- [Overview](#overview)
- [Creating your environment](#creating-your-environment)
- [Usage](#usage)
- [Simulator output](#simulator-output)
- [Simulator data](#simulator-data)
- [Contributing](#contributing)

## Overview

A [recent study](https://dl.acm.org/citation.cfm?id=3037698) has shown that power usage and latency of inference by deep AI models can be minimized if the model is split into two parts: 
- One that runs on the mobile device
- The other that runs in the cloud

![DFTS image](_static/img/collaInt.png)

Our simulator is developed in Python to run with Keras models. The user can choose a keras model and specify the following:
- Layer at which the model is split
- The following transmission parameters(currently supported):
  - n-bit quantization
  - channel
  - error concealment techniques
  
## Creating your environment

First clone this repository onto your local machine.

```bash
git clone https://github.com/SFU-Multimedia-Lab/DFTS.git
```

Create and activate a virtual environment on your machine, and navigate to the directory containing the repository.

Install the required dependencies.
```bash
pip install -r requirements.txt
```
If faced with any problems, contents after '==' after each library name can be deleted and run again.

## Usage

The main components with which the user interacts, includes the configuration files:
- params.yml
- taskParams.yml

After initializing these with the desired configurations, run

```bash
python main.py -p params.yml
```

The params configuration file consists of the following:

| Parameter | Description | Example |
| --- | --- | --- |
| **Task(integer)** | <ul><li> value: the task the model is designed for</li> <li> epochs: number of times the monte carlo simulation is run</li></ul> | <ul><li>0 classification, 1 object detection</li> <li>Any integer value </li></ul> |
| **TestInput** | <ul><li>dataset: dataset in use</li><li>batch_size: integer denoting the number of samples per batch</li><li>testdir:<ul><li>images: list containing the path to the directory of test images</li><li>annotations: list containing paths to annotations directory, for object detection only</li><li>testNames: list containing paths to text files containing names of the images</li></ul></li></ul> | <ul> <li>imagenet</li> <li>8</li> <li>testdir:<ul><li>['../annoDirs']</li><li>['../imageDirs']</li><li>['../test.txt']</li></ul></li></ul> |
| **Model** | <ul><li>kerasmodel: official keras model or path to h5 file containing both weights and architecture</li><li>customObjects: custom modules, classes, functions used to construct the model</li></ul> | <ul> <li>vgg16 or '../model.h5'</li> <li>keras_layers.example, for each list</li></ul> |
| **SplitLayer** | Layer at which the model is split, must be one of the names used for the layers | block1_pool, in the case for vgg16 |
| **Transmission** | <ul><li>rowsperpacket: number of rows of the feature map to be considered as one packet</li><li>quantization: number of bits and bool value indicating whether this paramter is to be included for this simulation</li><li>channel: a channel is selected by providing a bool for the include parameter for that channel, corresponding channel parameters are provided</li><li>concealment: technique for packet loss concealment is chosen by providing a bool value in the include paramter for that technique, corresponding paramters must be provided</li></ul> | <ul><li>8, True</li><li> to select randomLossChannel: 0, True</li></li><li>to select linear concealment, include: True</li></ul> | 
| **OutputDir** | directory where the results of the simulation must be stored | '../simData' |

The taskParams configuration file consists of the following paramters for each selected task:

- reshapeDims: list denoting the reshape dimensions of the images
- num_classes: integer denoting the number of classes in the dataset
- metrics: a dictionary containing the metrics the model needs to be evaluated against

Currently, only the parameters provided in the configuration files are supported. The simulation will break if any attempt is made to change the name of the parameters.

Sample configuration files for classification and object detection are provided in the sample folder.

Examples on how to organize data for input to the simulator can be found as follows:

* [Classification](https://drive.google.com/open?id=1R4Zn6UaZlADqHeKM7ij1k-G-STPkoPN_)
* [Object Detection](https://drive.google.com/open?id=1oBTJ_fIZKwfLfY4Ew5ZPwEPtGkMdYRgh)

A small subset of these images can be found in the repository itself. First switch to the test-images branch by executing the following:

```bash
git checkout test-images
```

Navigate to the sampleImages folder contained in the sample directory.

## Simulator output

The simulator outputs timing and indexing information to the terminal.

## Simulator data

The data produced by the simulator will be stored in the specified directory, as a numpy array in a **.npy** file. The name of the numpy file reflects the parameters of the simulation.

For example if the following parameters are used:

- splitlayer: block1_pool
- gilbert channel with 10 percent loss and 1 burst length
- 8 bit quantization
- concealment included

The resulting file name is:
```
block1_pool_8BitQuant_EC.npy
```

## Contributing

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussions.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.
