# Assignment 4

## Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#table-of-contents)
- [Experiment 1](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment_1)
- [Experiment 2](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment_2)
- [Experiment 3](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment_3)
- [Experiment 4](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment_4)
- [Experiment 5](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment_5)

## Experiment 1

Target:

1. Get the complete setup right.
2. Result:
    - Parameters: 35,312
    - Best Train Accuracy: 99.59
    - Best Test Accuracy: 99.05
3. Analysis:
    - The model is bulky
    - The model is not performing really well as the test accuracy is low.
    - 

## Experiment 2

1. Target: 
    - Create a light weight model.
2. Result:
    - Parameters: 16,695
    - Best Train Accuracy: 99.37
    - Best Test Accuracy: 98.95
3. Analysis:
    - The model is highly overfitted.
    - The model is stuck at a accuracy point. It didn't learn more features from training data. 
    - Increasing model capacity can help


## Experiment 3

1. Target:
    - First fix overfitting.
    - Add Regularization (Batch normalization) to every layer except the last.
    - Added GAP to convert 2D data to 1D data 

2. Result:
    - Parameters: 9,845
    - Best Train Accuracy: 99.49
    - Best Test Accuracy: 99.32
3. Analysis:
    - The model is not much overfitted.
    - GAP further reduced the parameter. Accuracy


## Experiment 4

1. Target:
    - Increase model's capacity by increasing data augmentation.
    - Also added dropout to further reduce the overfitting.

2. Result:
    - Parameters: 7,680
    - Best Train Accuracy: 
    - Best Test Accuracy: 
3. Analysis:
    - 
    - 


## Experiment 5

1. Target:
    - Model reached to optimal parameter and capacity
2. Result:
    - Parameters: 7,680
    - Best Train Accuracy: 98.99%
    - Best Test Accuracy: 99.5%
3. Analysis:
    - The model is with reduced parameters, not overfitted
    - LR scheduling helped the model converge better to optimal global minima.
