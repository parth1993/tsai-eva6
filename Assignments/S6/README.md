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
```
Batch Normalization (BN) normalizes the mean and standard deviation for each individual feature channel/map.
```
- In each layer, the input images/features will be equal to the batch size.
- Each such image/feature will be collection of channels.
- From a mathematical point of view, you can think of it as bringing the features of the image in the same range.
![image](https://user-images.githubusercontent.com/46129975/121207853-3a613c80-c897-11eb-9493-ba6e6cf36324.png)
- The size of feature maps is N × C × H × W N \times C \times H \times WN×C×H×W (N = 4 N = 4N=4 in this example). 4D tensor (N × C × H × W N \times C \times H \times WN×C×H×W), indicating number of samples, number of channels, height and width of a channel respectively.
