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

![formula](https://render.githubusercontent.com/render/math?math=\Large%20(i)\quad20\mu_{\mathcal{B}}%20\leftarrow%20\frac{1}{m}%20\sum_{i=1}^{m}%20x_{i})

- This first operation calculates the mean of the inputs within a mini-batch. 
- The result of the operation is a vector that contains each input’s mean.
    - ‘m’ refers to the number of inputs in the mini-batch.
    - ‘µ’ refers to the mean.
    - ‘B’ is a subscript that refers to the current batch.
    - ‘xi’ is an instance of the input data.
    - The mean(‘µ’) of a batch(‘B’) is calculated by the sum of the several input instances of the batch and dividing it by the total number of inputs(‘m’).

![formula](https://render.githubusercontent.com/render/math?math=\Large%20(ii)\quad\sigma_{\mathcal{B}}^{2}%20\leftarrow%20\frac{1}{m}%20\sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2})

