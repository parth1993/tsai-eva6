# Assignment 6 - Late Assignment on Time

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S7/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#table-of-contents)
- [About Our Code](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#about-our-code)
- [Convolution & Types](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#normalization--types)
    - [Dilation Convolution](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#1-batch-normalization)
    - [DepthWise Separable Convolution](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#2-layer-normalization)
- [Graphs for Our Model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#graphs-for-models-with-different-normalization-and-regularization)
    - [Training Loss per Epoch](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#1-training-loss)
    - [Training Accuracy per Epoch](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#3-training-accuracy)
    - [Test Loss per Epoch](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#2-test-loss)
    - [Test Accuracy per Epoch](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#4-test-accuracy)
- [Our Finidings for different Convolution Techniques](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#our-finidings-for-different-normalization-techniques)
- [Misclassified Images for each model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#misclassified-images-for-each-model)
    
# About Our Code
- [S6_Normalization_Techniques.ipynb](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S7/S7_Convolution_Techniques.ipynb) file performs the following:
    - Loads cifer data and creates train and test dataloader
    - Imports model from model.py
    - Defines train, test and eval loops
    - Define three models with different optimizers, loss and normalization.
    - Train three different models with different params
    - Collect accuracy and losses for all three and plot evaluation actual vs predicted images (for wrongly classified mnist class).
    - Plot training and test losses and accuracies for three different models
- [model.py](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S7/model.py) file performs the following:
    - Defines a Net() class. It defines a classification model with multiple conv blocks.


# Convolution & Types
## 1. Dilation Convolution
- A way of increasing the receptive field.
- An alternative to max pool.
- It gives broader perspective by integrating knowledge from the wider context. Like locating all similar features from image. Hence must be used with 3x3
- Helps in reduction of size of channel
## 2. Depthwise Separable Convolution
-

# Graphs for Model
## 1. Training Loss
![image](https://user-images.githubusercontent.com/46129975/121721414-dd1cf380-cb01-11eb-9736-f061e4d3eace.png)

## 2. Test Loss
![image](https://user-images.githubusercontent.com/46129975/121721348-cb3b5080-cb01-11eb-992d-23ba3751d8a8.png)

## 3. Training Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721399-d8583f80-cb01-11eb-98fc-56b7ab668685.png)

## 4. Test Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721369-d0989b00-cb01-11eb-8815-3ecaa70148cc.png)

# Model approach and takeaways:
- Initial model with standard convolution was giving the target accuracy of 85% but the number of parameters were high(500k parameters).
- Using Depthwise Separable Convolution helped reduced the parameters and more fine tuning the model helped achieved the accuracy with more number of epochs.
- Depthwise Separable Convolutions are an easy way to reduce the number of trainable parameters in a network at the cost of a small decrease in accuracy.

# Misclassified Images for model
![image](https://user-images.githubusercontent.com/46129975/121723564-4a318880-cb04-11eb-89d5-8fba9584c59e.png)
