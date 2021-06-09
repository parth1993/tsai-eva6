# Assignment 6

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S6/README.md#group-members)





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

# Grpahs for Models with different Normalization and Regularization
## 1. Training Loss
![image](https://user-images.githubusercontent.com/46129975/121379977-26ced800-c962-11eb-9aec-c8503ab899fc.png)

## 2. Test Loss
![image](https://user-images.githubusercontent.com/46129975/121380055-30584000-c962-11eb-811c-a76afb1c8175.png)

## 3. Training Accuracy
![image](https://user-images.githubusercontent.com/46129975/121380153-45cd6a00-c962-11eb-9e23-8eccd31f678f.png)

## 4. Test Accuracy
![image](https://user-images.githubusercontent.com/46129975/121380096-3a7a3e80-c962-11eb-9aeb-374770c436bf.png)

# Misclassified Images for each model
## 1. Network with Group Normalization + L1
![image](https://user-images.githubusercontent.com/46129975/121379565-c5a70480-c961-11eb-9a43-fe1112deaccc.png)

## 2. Network with Layer Normalization + L2
![image](https://user-images.githubusercontent.com/46129975/121379629-d6577a80-c961-11eb-9be1-870aab334ddf.png)

## 3. Network with L1 + L2 + BN
![image](https://user-images.githubusercontent.com/46129975/121379905-17e82580-c962-11eb-929f-4e1909d98216.png)
