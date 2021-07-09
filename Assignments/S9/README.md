# Resnets_and_Higher_Receptive_Fields

## Late Assignment on Time

### [Link to repo for files like utils, main, etc,.](https://github.com/amanjain487/CIFAR_10)
### [Link to Colab file on Github](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/Resnets_and_Higher_Receptive_Fields.ipynb)

## Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#table-of-contents)
- [Objective](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#objective)
- [Parameters and Hyperparameters](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#parameters-and-hyperparameters)
- [Results](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#results)
- [Graphs for Losses and Accuracies](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#graphs-for-losses-and-accuracies)
- [Correctly Classified Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S9/README.md#correctly-classified-images)

## Objective
- Train for exactly 24 Epochs
- Finf Lr_min and lr_max
- Apply these transforms while training:
  - RandomCrop(32, padding=4)
  - Flip Lr
  - CutOut(8x8)
- Must use One Cycle Policy

## Parameters and Hyperparameters
- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Scheduler: One Cycle Policy
- Batch Size: 512
- Epochs: 24

## Results
- Best Accracy is 89.17% at 39th Epoch.

## Graphs for Losses and Accuracies
![image](https://user-images.githubusercontent.com/46129975/124268705-7e262980-db57-11eb-831e-fa0f1ab2c045.png)
