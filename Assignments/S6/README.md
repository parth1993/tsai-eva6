# Assignment 6 - Late Assignment on Time

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S6/README.md#group-members)
- [Table of Contents]()
- [Normalization & Types]()
    - [Batch Normalization]()
    - [Layer Normalization]()
    - [Group Normalization]()
- [Our Finidings for different Normalization Techniques]()
- [Grpahs for Models with different Normalization and Regularization]()
    - [Training Loss per Epoch]()
    - [Training Accuracy per Epoch]()
    - [Test Loss per Epoch]()
    - [Test Accuracy per Epoch]()
- [Misclassified Images for each model]()
    - [Network with Group Normalization + L1]()
    - [Network with Layer Normalization + L2]()
    - [Network with L1 + L2 + BN]()

# Normalization & Types
- In general, a normalization layer will try to mean-center and make feature maps have unit-variance.
- There are three types of Normalization
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

## 1. Batch Normalization
- Batch Normalization will compute two scalars,  μ  and  σ^2, for each channel. 
- For example, as shown in the following figure: 
    - Each colored group of cells results in one scalar  μ  and one scalar  σ^2. In other words, one pair of  (μ,σ^2) from all the blue cells, one pair of  (μ,σ^2) from all the orange cells and same for green and purple.
- So, **total number of (μ,σ^2) pairs** will be **equal to** the **total number of channels, C**.
![image](https://user-images.githubusercontent.com/46129975/121216175-4997b880-c89e-11eb-8126-931f95d21ed8.png)
- For example, 
![image](https://user-images.githubusercontent.com/46129975/121217318-5bc62680-c89f-11eb-93f1-787166c13b37.png)

## 2. Layer Normalization
- Layer Normalization will compute two scalars  μ  and  σ^2  for each sample. 
- Therefore, **total number of (μ,σ^2) pairs** will be **equal to** the **total number of samples, N**.
![image](https://user-images.githubusercontent.com/46129975/121216135-40a6e700-c89e-11eb-8131-3951eeea3c6e.png)
- For example,
![image](https://user-images.githubusercontent.com/46129975/121218959-dba0c080-c8a0-11eb-8ce6-49e00a92a770.png)


## 3. Group Normalization
- Group Normalization with a group size  **g**, will group the channels into multiple groups, and then computes two scalars  μ  and  σ^2 for each sample and each group. 
- For this example, we have  4  channels, and group size is  g=2. 
- Therefore, there are  2  groups of channels ( groups=2 ), and as a result: **total number of (μ,σ^2) pairs** will be **equal to N*C/g**.
![image](https://user-images.githubusercontent.com/46129975/121220943-b7de7a00-c8a2-11eb-9b92-0cc728c7d307.png)
- For example,
![image](https://user-images.githubusercontent.com/46129975/121220796-941b3400-c8a2-11eb-828c-0a8fb94a8b4e.png)

# Our Finidings for different Normalization Techniques
- We normalize the data so that calculations are smaller and also to make features common.
    - Like without normalization ear of black cat is very much different than ear of white cat.
    - By normalization we try to remove the color and focus only on ear in some channel.
- Batchnorm is better because it works feature wise.
    - Layer norm and groupnorm work image wise and they actually suppress important feature maps based on useless feature maps.
    - So it will be difficult to assign weight to such feature map which can vary from image to image. 
    - Whereas in batchnorm if a feature map is normalized, it is normalized for all images, and backprop will take care in determining the importance of that feature.
    - 
# Grpahs for Models with different Normalization and Regularization
## 1. Training Loss
![image](https://user-images.githubusercontent.com/46129975/121721414-dd1cf380-cb01-11eb-9736-f061e4d3eace.png)
## 2. Test Loss
![image](https://user-images.githubusercontent.com/46129975/121721348-cb3b5080-cb01-11eb-992d-23ba3751d8a8.png)
## 3. Training Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721399-d8583f80-cb01-11eb-98fc-56b7ab668685.png)
## 4. Test Accuracy
![image](https://user-images.githubusercontent.com/46129975/121721369-d0989b00-cb01-11eb-8815-3ecaa70148cc.png)
# Misclassified Images for each model
## 1. Network with Group Normalization + L1
![image](https://user-images.githubusercontent.com/46129975/121720963-5536e980-cb01-11eb-817e-1721721c3c0f.png)

## 2. Network with Layer Normalization + L2
![image](https://user-images.githubusercontent.com/46129975/121721031-6a137d00-cb01-11eb-9723-88e2b4d0592c.png)

## 3. Network with L1 + L2 + BN
![image](https://user-images.githubusercontent.com/46129975/121721116-84e5f180-cb01-11eb-86c8-5ac78856ef72.png)
