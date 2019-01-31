# shapenet

This repository contains the [PyTorch](https://pytorch.org) implementation of [our Paper](#our-paper)

## Contents
* [Installation](#installation)
* [Usage](#usage)
  * [By Scripts](#by-scripts)
  * [From Python](#from-python)
  * [Pretrained Weights](#pretrained-weights)
 * [Our Paper](#our-paper)

## Installation
Currently only the installation from source is supported, which can be done by `pip install git+https://github.com/justusschock/shapenet` 

## Usage
### By Scripts
For simplicity we provide several scripts to preprocess the data, train networks, predict from networks and export the network via [`torch.jit`](https://pytorch.org/docs/stable/jit.html).
To get a list of the necessary and accepted arguments, run the script with the `-h` flag.

#### Data Preprocessing
* `prepare_all_data`: prepares multiple datasets (you can select the datasets to preprocess via arguments passed to this script)
* `prepare_cat_dset`: Download and preprocesses the [Cat-Dataset](https://www.kaggle.com/crawford/cat-dataset)
* `prepare_helen_dset`: Preprocesses an already downloaded ZIP file of the [HELEN Dataset](http://www.ifp.illinois.edu/~vuongle2/helen/) (Download is recommended from [here](https://ibug.doc.ic.ac.uk/download/annotations/helen.zip) since this already contains the landmarks)
* `prepare_lfpw_dset`: Preprocesses an already downloaded ZIP file of the [LFPW Dataset](https://neerajkumar.org/databases/lfpw/) (Download is recommended from [here](https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip) since this already contains the landmarks)

#### Training
* `train_shapenet`: Trains the shapenet with the configuration specified in an extra configuration file (exemplaric configuration for all avaliable datasets are provided in the [example_configs](example_configs) folder)

#### Prediction
* `predict_from_net`: Predicts all images in a given directory (assumes existing groundtruths for cropping, otherwise the cropping to groundtruth could be replaced by a detector)

#### JIT-Export
* `export_to_jit`: Traces the given model and saves it as jit-ScriptModule, which can be accessed via Python and C++

### From Python
This implementation uses the [`delira`-Framework](https://github.com/justusschock/delira) for training and validation handling. It supports mixed precision training and inference via [NVIDIA/APEX](https://github.com/NVIDIA/apex) (must be installed separately). The data-handling is outsourced to [shapedata](https://github.com/justusschock/shapedata).

The following gives a short overview about the packages and classes.

#### `shapenet.networks` 
The `networks` subpackage contains the actual implementation of the shapenet with bindings to integrate the `ShapeLayer` and other feature extractors (currently the ones registered in `torchvision.models`).

#### `shapenet.layer`
The `layer` subpackage contains the Python and C++ Implementations of the ShapeLayer and the Affine Transformations. It is supposed to use these Layers as layers in `shapenet.networks`

#### `shapenet.jit`
The `jit` subpackage is a less flexible reimplementation of the subpackages `shapenet.networks` and `shapenet.layer` to export trained weights as jit-ScriptModule

#### `shapenet.utils`
The `utils` subpackage contains everything that did not suit into the scope of any other package. Currently it is mainly responsible for parsing of configuration files.

#### `shapenet.scripts`
The `scripts` subpackage contains all scipts described in [Scripts](#by-scripts) and their helper functions.

### Pretrained Weights
**Coming Soon**

## Our Paper
**Coming Soon**
