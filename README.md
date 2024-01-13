# Machine Learning for Tsunamis

## Introduction

This repository contains machine learning tools to forecast results from tsunami simulations. They have been implemented by the EDANYA Group at the University of Málaga, Spain. These tools use the [EDDL library](https://github.com/deephealthproject/eddl).

CMake files are provided to compile all the codes. The executables can be run without arguments to see the use format.

The training codes include an early stopping strategy and the reduction of the learning rate on plateau, which can be configured. The network architecture, maximum number of epochs, batch size, and the initialization mode of the training, validation and test sets can also be defined by the user.


## Requirements

EDDL (C++ version)

An MPI library


## Alert Levels

The `alert_levels` folder corresponds to the classification problem. It contains codes to train and infer tsunami alert levels. Specifically, four alert levels are considered: green (very low), yellow (moderate), orange (high) and red (very high).

### Training

The `src_train` folder contains the source code for the training of the models. It uses MPI to run multiple trainings in parallel. At the end of the execution, the best obtained model, which is selected based on the results achieved in the validation set, is saved in ONNX format.

The `use_cpu` variable (in line 39 of `alert_levels.cxx` file) can be set to true or false to perform the trainings on CPU or GPU.

At the beginning of the `eddl.cxx` file, the number of GPUs per node, the ranges of the Okada parameters, and the loss function (categorical cross entropy or accuracy) can be defined.

### Inference

The `src_infer` folder contains the source code for the inference of results using a previously trained model.

At the beginning of the `eddl.cxx` file, the ranges of the Okada parameters can be defined. They should be the same that were used in the training process.


## Maximum Water Height

The `max_height` folder corresponds to the regression problem. It contains codes to train and infer the maximum water height of a tsunami.

### Training

The `src_train` folder contains the source code for the training of the models. It uses MPI to run multiple trainings in parallel. At the end of the execution, the best obtained model, which is selected based on the results achieved in the validation set, is saved in ONNX format.

The `use_cpu` variable (in line 40 of `max_height.cxx` file) can be set to true or false to perform the trainings on CPU or GPU.

At the beginning of the `eddl.cxx` file, the number of GPUs per node and the ranges of the Okada parameters can be defined. The mean squared error (MSE) is used as the loss function.

At the end of the training process, a file `inference_parameters.txt` is generated. This file should be passed as argument to the inference executable, in order to perform the same normalization of the output layer in the training and inference processes.

### Inference

The same considerations made in the classification problem apply here.


## File formats

The inputs and outputs of the network should be defined in two text files. The format of the input file is the same in the classification and regression problems.

The input file has one line for each sample, where each sample is defined by its nine Okada parameters:

`<longitude(degrees) latitude(degrees) depth(km) fault_length(km) fault_width(km) strike(degrees) dip(degrees) rake(degrees) slip(m)>`

The output of the network for the classification problem consists of one line per sample, where each line is a one-hot vector having four elements, with the element corresponding to the alert level of the sample equal to one, and the rest of elements equal to zero. For example, the vector `0 0 1 0` represents an orange level.

The output of the network for the regression problem consists of one line per sample, where each line has one or several values of maximum water heights, one value for each point that the user wants to consider.


## License

These EDDL codes are distributed under GNU GENERAL PUBLIC LICENSE Version 2.


## Links

EDANYA group: [https://www.uma.es/edanya](https://www.uma.es/edanya)

EDDL: [https://github.com/deephealthproject/eddl](https://github.com/deephealthproject/eddl)
