# Assignment 7 - Late Assignment on Time

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S7/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#table-of-contents)
- [Data Augmentation Techniques Used](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7/#Data-Augmentation-Techniques-Used)
- [Convolution & Types](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#Convolution-&-Types)
- [Graphs for Our Model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7/#Graphs-for-Model) 
- [Model approach and takeaways](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#Model-approach-and-takeaways)
- [Misclassified Images for each model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#Misclassified-images)
    

# Data Augmentation Techniques Used
- HorizontalFlip - Flip the input horizontally around the y-axis.
- ShiftScaleRotate - Randomly apply affine transforms: translate, scale and rotate the input.
- CoarseDropout - the technique of removing many small rectanges of similar size.
- ToGray - Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.

# Convolution & Types
## 1. Dilation Convolution
![image](https://user-images.githubusercontent.com/16293041/122604966-68fbc600-d094-11eb-80b2-c4a220865ee0.png)
- A way of increasing the receptive field.
- An alternative to max pool.
- It gives broader perspective by integrating knowledge from the wider context. Like locating all similar features from image. Hence must be used with 3x3
- Helps in reduction of size of channel
## 2. Depthwise Separable Convolution
![image](https://user-images.githubusercontent.com/16293041/122604869-42d62600-d094-11eb-9594-aef642247f9c.png)
- Split the input into channels, and split the filter into channels (the number of channels between input and filter must match).
- For each of the channels, convolve the input with the corresponding filter, producing an output tensor (2D).
- Stack the output tensors back together.


# Graphs for Model
## Training Loss, Test Loss, Training Accuracy, Test Accuracy
![graphs](https://user-images.githubusercontent.com/16293041/122614291-00b4e080-d0a4-11eb-9b4e-c83986c8b59e.png)


# Model approach and takeaways:
- Initial model with standard convolution was giving the target accuracy of 85% but the number of parameters were high(500k parameters).
- Using Depthwise Separable Convolution helped reduced the parameters and more fine tuning the model helped achieved the accuracy with more number of epochs.
- Depthwise Separable Convolutions are an easy way to reduce the number of trainable parameters in a network at the cost of a small decrease in accuracy.

# Misclassified images
![misclassified](https://user-images.githubusercontent.com/16293041/122615888-f811d980-d0a6-11eb-9bbc-4838fa56da6b.png)
