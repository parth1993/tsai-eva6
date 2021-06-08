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
- $$
\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i}
$$
![image](https://user-images.githubusercontent.com/46129975/121200455-611c7480-c891-11eb-9dca-1ab1ab5be7d5.png)

