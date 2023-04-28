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

&emsp;For this project, we will be utilizing the MNIST database (Modified National Institute of Standards and Technology database). This is a large database of 28 x 28 grayscale pixel handwritten images of numbers. The dataset contains 60,000 training images and 10,000 testing images. In their original paper, they achieve an error rate of 0.8% using a support vector machine. Although our feedforward neural network does not have to have accuracy comparable to the original research, it does ensure there are minimal upper bounds on the performance of the model due to the nature of the dataset.

![image](https://user-images.githubusercontent.com/80783579/235015094-23583509-d380-4efe-a778-ed798e35efec.png)
<br /><br />
&emsp;For this project there will be two sessions of benchmarking. All benchmarks will be performed against a PyTorch neural network with two layers, the first will be a fully connected layer with the input being the image as a flattened vector (length 784). The first layer will output 512 features, which are inputs into a second fully connected layer which outputs features. The 10 outputs of the second layer are softmaxed to the integer (0-9) prediction confidence percentages, and the prediction of most confidence will be the model’s output. Each training and testing epoch will iterate through the entire MNIST dataset every time. To limit code runtime, I am limiting the model to 5 epochs, as later creating adversarial images and training / testing against them takes a substantial amount of time.

![image](https://user-images.githubusercontent.com/80783579/235015002-bb82f8de-a69e-47e4-bfa8-8f08b05c3212.png)
<br /><br />
&emsp;The first benchmarking iteration will be training with the MNIST training dataset. It will be tested against the original testing dataset, the testing dataset altered using the fast-gradient method (FGM), and the testing dataset altered using projected gradient descent (PGD). The second benchmarking iteration will be training with the MNIST training dataset AND the adversarial attack images, meaning that the model should provide better results against attacks. The testing will be the same as the first benchmark. Both attacks will be implemented using the CleverHans Python library which provides support for basic attacks against PyTorch neural networks.

### Results

&emsp;The first round of benchmarking returned massive blows to the neural network due to adversarial attacks. After 5 epochs taking a total of 3 minutes and 24 seconds, the model tested with a 0.918% accuracy on the base testing database images, however performed with a 0.189% accuracy on the FGM images and 0.140% accuracy on the PGD images. This substantial decrease suggests that this model is incredibly vulnerable to these attacks, as these attacks render the model’s performance to essentially guessing levels. Additionally, we see test loss increase from 0.0047 in the base testing to 0.0409 against FGM images and 0.0465 against PGD images.

![image](https://user-images.githubusercontent.com/80783579/235042298-d445d35e-ad8f-4aff-8d24-535ffd925239.png)
![image](https://user-images.githubusercontent.com/80783579/235042333-6a88fd44-e757-4eab-8866-10d51882fb69.png)
<br /><br />
&emsp;Retraining with the adversarial cases in mind took substantially more time, as the code had to train on the original image, the FGM affected image, and the PGD affected image for each image in the testing dataset every epoch. The 5 epochs intotal took 14 minutes to train. The performance improvements were substantial however, as the model actually performed better on the base images with the additional training at 0.943% accuracy, while achieving 0.739% accuracy against FGM images and 0.699% accuracy against PGD images. This is an encouraging result suggesting the adversarially trained model can handle attacks significantly better than the previous model at no cost to original model performance. Furthermore, loss for base testing decreased to 0.0032, while loss for FGM decreased to 0.0109 and loss for PGD decreased to 0.0122.

![image](https://user-images.githubusercontent.com/80783579/235042363-0f37786a-89e5-481a-af16-2025976d6872.png)
![image](https://user-images.githubusercontent.com/80783579/235042381-5ea1a28d-38d7-4c57-ae7d-84eea07364e3.png)
<br /><br />
&emsp;For the neural network, two feedforward layers were chosen to preserve simplicity within the model, avoiding error or loss that could be caused by a complicated model such as a convolutional neural network handling the attacks differently. However, testing adversarial attacks on other models should follow a similar guideline to this project. A max epoch of 5 was chosen to prevent unreasonable testing times (which reached 14 minutes on the second benchmark). Both attacks had an epsilon value of 0.1, which is the variation in the attack that is acceptable. PGD, which works with gradient descent through multiple iterations, was allowed 40 iterations with a 0.01 epsilon step between each iteration, giving us a good degree of certainty a minimum was reached.

### Discussion

&emsp;The results obtained from the model provide a good omen on the ability for neural network models to properly defend against adversarial attacks. The capabilities of both FGM and PGD are significant in destroying a model’s performance and thankfully we now see a streamlined way to better protect our model’s performance. However, the performance of the model post adversarial training was not up to par with other researchers, such as Tim Cheng in a similar Towards Data Science study got his base model to an accuracy of 98% on base datasets and his adversarially trained model to an accuracy of 90% on attacks. His model performance likely comes from using 2 convolutional layers and 3 fully connected layers, compared to my much simpler 2 fully connected layers. Although this model performance was not replicated to the same extent in this report, it only emphasized the capabilities that are possible for adversarial training with complex models such as convolutional neural networks.

### Conclusion

&emsp;In this project, I have provided a benchmark for how a feedforward neural network implemented with PyTorch performs against two adversarial attacks (FGM and PGD) executed with the CleverHans Python library, and provided a way to utilize the CleverHans library to retrain the model to significantly improve its performance against these attacks. These results should not only allow data scientists to view the devastating effects of adversarial attacks, but also provide them a clear guideline to implementing these adversarial methods into their model training to produce more robust models. Not only is the original performance of the model recuperable after adversarial training, but it may leave your model with better accuracy and lower loss on the original training dataset.

### References

- [CleverHans GitHub repository](https://github.com/cleverhans-lab/cleverhans)
- [MNIST Dataset reserach paper](https://ieeexplore.ieee.org/document/726791)
- [Sidhant Haldar's Gradient-based Adversarial Attacks : An Introduction](https://medium.com/swlh/gradient-based-adversarial-attacks-an-introduction-526238660dc9)
- [Tim Cheng's Adversarial Attack and Defense on Neural Networks in PyTorch](https://towardsdatascience.com/adversarial-attack-and-defense-on-neural-networks-in-pytorch-82b5bcd9171)
- [Prof. Zhang's DS4400 Lecture Notes](http://www.hongyangzhang.com/DS4400_Spring2023.html)
