# Assignment 4

## Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#table-of-contents)
- [Experiment 1](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/EVA6_Session5_CODE1.ipynb)
- [Experiment 2](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment-2)
- [Experiment 3](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment-3)
- [Experiment 4](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment-4)
- [Experiment 5](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/README.md#Experiment-5)

## Experiment 1(https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S5/EVA6-Session5-CODE1.ipynb)

1. Target:
    - Get the complete setup right.

2. Result:
    - Parameters: 35,312
    - Best Train Accuracy: 99.59%
    - Best Test Accuracy: 99.05%

3. Analysis:
    - The model is bulky
    - The model is not performing well as the test accuracy is low.


## Experiment 2

1. Target: 
    - Create a light weight model.

2. Result:
    - Parameters: 16,695
    - Best Train Accuracy: 99.37%
    - Best Test Accuracy: 98.95%

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
    - Best Train Accuracy: 99.49%
    - Best Test Accuracy: 99.32%

3. Analysis:
    - The model is still bit overfitted.
    - GAP further reduced the parameter. Accuracy goes down with GAP but the test accuracy finally goes up with regularizion.


## Experiment 4

1. Target:
    - Increase model's capacity by increasing data augmentation.
    - Also reduce the overfitting further by adding droput.

2. Result:
    - Parameters: 7,680
    - Best Train Accuracy: 98.96%
    - Best Test Accuracy: 99.4%

3. Analysis:
    - Model is not converging to global minima and accuracy is wiggling.
    - Small number of channels in the starting seems to not having full capacity of the model. We can experiment with larger number of channels in the beginning.
    - Model is good in terms of regularization.

## Experiment 5

1. Target:
    - Increase model's capacity more at the starting. Add more channels and try.
    - Try Converging the model to the optimum global minima.

2. Result:
    - Parameters: 7,384
    - Best Train Accuracy: 99.3%
    - Best Test Accuracy: 99.5%

3. Analysis:
    - The model is with reduced parameters, not overfitted
    - LR scheduling helped the model converge better to optimal global minima.
    - Increasing channels in the beginning helped model to learn better.
