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
- [Graphs for Models with different Normalization and Regularization](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S6#graphs-for-models-with-different-normalization-and-regularization)
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
- [model.py](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S6/model.py) file performs the following:
    - Defines a Net() class for different normalization. It takes in input param which is type of normalization and creates a classification with earlier best performing model with multiple conv blocks.
    - Defines a function to club conv layer with different normalization when normalization type is provided as argument.


# Convolution & Types
- In general, a normalization layer will try to mean-center and make feature maps have unit-variance.
- Here we are experimenting with below three types of Normalization
    - Batch Normalization
    - Layer Normalization
    - Group Normalization

- Let’s assume  h  is a feature map in a CNN, so that means  `h`  has four dimensions:  
    ```<batch;channel;width;height>``` 
- So we can represent  h  as a 4-dimensional tensor like in the following example with a batch of size  4  samples.
- Each sample having  4  feature maps of size  6×6.

![image](https://user-images.githubusercontent.com/46129975/121214933-320c0000-c89d-11eb-94b0-69a54ac5c9a3.png)

- Then, a normalization layer will compute the mean  μ  and variance σ^2  from the data, and then normalize the feature maps as follows:

![image](https://user-images.githubusercontent.com/46129975/121215129-5cf65400-c89d-11eb-9226-91606add200c.png)

- where
    - γ  is a scaling parameter,  
    - β  is a shift parameter,
    - ϵ  is to avoid numerical instability (division-by-zero problem).

- While this general formulation is the same among different normalization layer, the difference between them is the way  μ  and  σ  are computed, which is explained below.

- Advantages of Normalization : We normalize the data so that calculations are smaller and also to make features common.
    - Like without normalization ear of black cat is very much different than ear of white cat.
    - By normalization we try to remove the color and focus only on ear in some channel.

## 1. Dilation Convolution
- Batch Normalization will compute two scalars,  μ  and  σ^2, for each channel. 

# Graphs for Models 
## 1. Training Loss
![image](https://user-images.githubusercontent.com/46129975/121721414-dd1cf380-cb01-11eb-9736-f061e4d3eace.png)

## 2. Test Loss
![image](https://user-images.githubusercontent.com/46129975/121721348-cb3b5080-cb01-11eb-992d-23ba3751d8a8.png)

## 3. Training Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721399-d8583f80-cb01-11eb-98fc-56b7ab668685.png)

## 4. Test Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721369-d0989b00-cb01-11eb-8815-3ecaa70148cc.png)

# Model Analysis and approach
- Looking at the testing graph, Batch normalization results into better performance as compared to layer and Group normalization for our model. Reason can be the following:
    - BatchNorm works feature wise.
    - Layer norm and groupnorm work image wise and they actually suppress important feature maps based on useless feature maps.
    - So in case of layer and group norm, it will be difficult to assign weight to such feature map which can vary from image to image.
    - Whereas in BatchNorm, if a feature map is normalized, it is normalized for all images, and backprop will take care in determining the importance of that feature.
- Training time is bit lesser when used Normalization on data. It reduced calculations.
- Model has converged faster. From around 14th epoch, it shows constant results.

# Misclassified Images for model
## 1. Network with Group Normalization + L1
![image](https://user-images.githubusercontent.com/46129975/121723564-4a318880-cb04-11eb-89d5-8fba9584c59e.png)
