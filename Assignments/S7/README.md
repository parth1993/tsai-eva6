# Assignment 7 - Late Assignment on Time

# Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S7/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#table-of-contents)
- [About Our Code](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S7#about-our-code)
- [Data Augmentation Techniques Used]
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
