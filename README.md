# Pytorch-Intro
A long and comprehensive Introduction into the world of PyTorch for Deep Learning and Machine Learning


**File Pytorch Intro* (pytorch_Intro.py)
-----------------------------------------------------------------------------------------------------------------
This file is just a simple introduction into tensors using PyTorch as well as Numpy. The code shows a bunch of tensors that have been manipulated as well as some arrays and matrix multiplication methods that are fundamental in the learning of Maching/Deep Learning. Code portraying things such as:
  - tensors - a way to represent data
  - scalar
  - matrix
  - tensor
  - creating a random tensor
  - create random tensor with similar shape to image tensor
  - tensor datatypes
  - Manipulating Tensors
  - Matrix Multiplication aka dot product
  - Reshaping Tensors (add dimensions, subtract (squeeze))
  - torch.permute - rearranges the dimensions of a target tensor in specified order (used in images)
  - reproductability (taking the random out of random) start with random numbers-> tensor operations -> update random numbers to try and make them better representations of the data (repeat)

**First model using pytorch : Model_0* (pytorch_Intro.py)
-----------------------------------------------------------------------------------------------------------------
This file is the very first deep learning model I have made using pytorch. This code has a created linear data set that is then split up into training and testing data that is then visualized on a graph using matplotlib and then is converted into a linear regression model (predict the outcome basewd on the linear dataset) then the model is trained after setting up looss functions and optimizers to try and get an accuraate prediction:\
  - initialized weights and bias
  - split data into training and testing data
  - Vizualize the split data
  - create loinear regression model ( look at training data and adjust the random values, or better represent (get closer to) ideal values (aka weight and bias we used to create the data) by:
      # 1. Gradient Descent
      # 2. Back Propagation
  - create random seed - keeps the numbers consistent for reproducibility and consistent results
  - initialzie the model
  - use model to make predictions before training
  - training the model using a training loop
  - vizualize the results
