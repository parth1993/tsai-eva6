# Assignment 6

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S6/README.md#group-members)





# Types of Normalization
- There are three types of Normalization
    - Batch Normalization
    - Layer Normalization
    - Group Normalization

## 1. Batch Normalization
- Batch Normalization is done channel wise.
- In each layer, the input images/features will be equal to the batch size.
- Each such image/feature will be collection of channels.
- https://latex.codecogs.com/gif.latex?%24%5Cmu_%7B%5Cmathcal%7BB%7D%7D%20%5Cleftarrow%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20x_%7Bi%7D%24
![image](https://user-images.githubusercontent.com/46129975/121200455-611c7480-c891-11eb-9dca-1ab1ab5be7d5.png)

