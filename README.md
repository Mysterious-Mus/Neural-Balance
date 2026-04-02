This repository contains the code used in our experiments

The plots, code, and scripts for the experiments are divided by model type, and dataset into the following folders:

  FMNIST-FCN

    hist: folder containing the experimental results from our paper

    FashionMnist.py: contains the code used to run a single experiment on the fashion mnist dataset using a fully connected model architecture.
      Input arguments and description are located in the file, or can be referenced by adding '--help' at the end of the call to the script

    plotting.ipynb: contains the notebook used to plot experimental results
  
    Contains scripts to run code for each of the following experiments:
      runFmnistComparison.py: runs scripts for the balancing vs regularization vs plain methodology on the fashion mnist dataset using a fully connected model architecture.

  IMDB-RNN

    hist: folder containing the experimental results from our paper
    
    rnn_imdb.py: contains the code used to run a single experiment on the IMDB dataset using a recurrent neural network model architecture.
      Input arguments and description are located in the file, or can be referenced by adding '--help' at the end of the call to the script

    plotting.ipynb: contains the notebook used to plot experimental results
  
    Contains scripts to run code for each of the following experiments:
      runFBVsNoFBIMDB.py: runs scripts for both full balancing, and no full balancing performed at the start of training, tested on various training methodologies on the IMDB dataset using a recurrent neural network model architecture.
      runLimitedDataScripts.py: runs scripts for the balancing vs regularization vs plain methodology on the IMDB dataset using a recurrent neural network model architecture using a limited amount of train data.
  
  IMDB-Transformer

    transformer.py: contains the code used to run a single experiment on the IMDB dataset using a transformer model architecture.
      Input arguments and description are located in the file, or can be referenced by adding '--help' at the end of the call to the script

  MNIST-FCN

    hist: folder containing the experimental results from our paper
  
    Mnist.py: contains the code used to run a single experiment on the mnist dataset using a fully connected model architecture.
      Input arguments and description are located in the file, or can be referenced by adding '--help' at the end of the call to the script

    plotting.ipynb: contains the notebook used to plot experimental results
  
    Contains scripts to run code for each of the following experiments:
      runMnistComparison.py: runs scripts for the balancing vs regularization vs plain methodology on the mnist dataset using a fully connected model architecture.
      runFbVsNoFbMnist.py: runs scripts for both full balancing, and no full balancing performed at the start of training, tested on various training methodologies on the mnist dataset using a fully connected model architecture.
      runMnistFract.py: runs scripts for the balancing vs regularization vs plain methodology on the mnist dataset using a fully connected model architecture using a limited amount of train data.
      runFBVsNoFBMnistTanh.py: runs scripts for both full balancing, and no full balancing performed at the start of training, tested on various training methodologies on the mnist dataset using a fully connected model architecture with Tanh activation.

  toyExperiments

    toy_data_experiment.ipynb: Contains the notebookk used to generate the toy experiments in our proof of concept
