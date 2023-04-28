# DS4400 Final Project: Benchmarking Adversarial Attacks Against Torch Models

### Overview

&emsp;This repository provides my submission for the final project in Professor Zhang's DS4400 Machine Learning and Data Mining I class. In this work, I will implement two adverserial attacks (fast gradient methos and projected gradient descent) onto a PyTorch feedforward neural network designed to classify images from the MNIST dataset. I will benchmark the effects that the adverserial attacks have on accuracy and loss at testing and benchmark improvements from adverserial training.

### Introduction

&emsp;Adversarial attacks pose a threat to neural networks as it allows malicious actors to purposefully create wrong classifications. As neural networks are further implemented in vital societal tools such as automatic driving, facial recognition, and fraud detection, it is important that these models are tamper-proof to prevent costly or potentiall life-threating errors. Defending against adversarial attacks is unintuitive as it is a sharp turn from the standard in machine learning where the data you train and test against is clean and reliable. The experiment below should provide a replicatable path to training against adverserial attacks in any neural network.

![image](https://user-images.githubusercontent.com/80783579/235015257-134d8805-d317-45c8-95eb-29d550e4ecb7.png)
<br /><br />
&emsp;In order to measure the effects of how PyTorch models hold up against adversarial attacks, I will implement a two-layer feedforward neural network that can accurately classify grayscale images of handwritten numbers from the MNIST dataset; then, I will utilize the Python CleverHans library to run adversarial attacks against this feedforward neural network to benchmark the effect that these attacks have on the base model; finally, I will re-train the model with these adversarial inputs in mind to test how well the model can perform if we adjust for adversarial attacks ahead of time. The benefits of this approach are that the MNIST image classification problem is well known and has clean data, meaning that any large movements in accuracy or loss can be attributed to the adversarial attack rather than issues with the underlying data. Ideally our results should show the large impact an adversarial attack has on model performance and additionally an improvement when we train the model against these attacks.
<br /><br />
The adverserial attacks that will be used are white-box attacks, meaning that the attack takes in a normal input image and the neural network model with all the weights known to produce an adversarial image. Fast gradient method (FGM) works by solbing for the element which maximizes the cost for every element in the model. It does this in one iteration and is computationally efficient. Projected gradient descent (PGD) utilizes gradient descent to locate the point of highest loss in the model. This descent takes multiple iterations and is more computationally expensive. 

### Setup

&emsp;For this project, we will be utilizing the MNIST database (Modified National Institute of Standards and Technology database). This is a large database of 28 x 28 grayscale pixel handwritten images of numbers. The dataset contains 60,000 training images and 10,000 testing images. In their original paper, they achieve an error rate of 0.8% using a support vector machine (http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). Although our feedforward neural network does not have to have accuracy comparable to the original research, it does ensure there are minimal upper bounds on the performance of the model due to the nature of the dataset.

![image](https://user-images.githubusercontent.com/80783579/235015094-23583509-d380-4efe-a778-ed798e35efec.png)
<br /><br />
&emsp;For this project there will be two sessions of benchmarking. All benchmarks will be performed against a PyTorch neural network with two layers, the first will be a fully connected layer with the input being the image as a flattened vector (length 784). The first layer will output 512 features, which are inputs into a second fully connected layer which outputs features. The 10 outputs of the second layer are softmaxed to the integer (0-9) prediction confidence percentages, and the prediction of most confidence will be the model’s output. Each training and testing epoch will iterate through the entire MNIST dataset every time. To limit code runtime, I am limiting the model to 5 epochs, as later creating adversarial images and training / testing against them takes a substantial amount of time.

![image](https://user-images.githubusercontent.com/80783579/235015002-bb82f8de-a69e-47e4-bfa8-8f08b05c3212.png)
<br /><br />
&emsp;The first benchmarking iteration will be training with the MNIST training dataset. It will be tested against the original testing dataset, the testing dataset altered using the fast-gradient method (FGM), and the testing dataset altered using projected gradient descent (PGD). The second benchmarking iteration will be training with the MNIST training dataset AND the adversarial attack images, meaning that the model should provide better results against attacks. The testing will be the same as the first benchmark. Both attacks will be implemented using the CleverHans Python library which provides support for basic attacks against PyTorch neural networks.

### Results


