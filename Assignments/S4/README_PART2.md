# Assignment 4 - Architectural Basics
This part aims at designing a CNN model for MNIST Classification with below mentioned targets.

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_PART2.md#group-members)
- [Target]()
- [Part-2 - REWRITE COLAB CODE](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_PART2.md)
- [Model Architecture](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_PART2.md#neural-network)
- [Fetching MNIST Dataset and Dataset Prep](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_PART2.md#Fetching-Mnist-Dataset-and-Dataset-Prep)
- [Defining the Model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Defining-the-model)
- [Defining Model Object, LR Scheduler and Optimizer](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Defining-model-object,-loss-function-and-optimizer)
- [Train and Test the Model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Train-and-Test-the-Model)
- [Plotting Graphs for Losses and Accuracies](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Plotting-Graphs-for-Losses-and-Accuracies)
- [Custom Testing](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Custom-Testing)

## Group Members
- Harshita Sharma
- Pankaj S K
- Avichal Jain
- Aman Kumar

## Target
```
- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- Must have used Batch Normalization, Dropout, a fully connected layer and a global averaging pool layer.
- Any concept that has been covered can be used.
```

## Model Architecture
The model that is used to achieve the above targets is.
![2021-05-28](https://user-images.githubusercontent.com/16293041/119993770-90a5c400-bfe9-11eb-8afe-a164e5f7d34b.jpg)
We have stopped at Receptive Field of 20, because almost all the images in MNIST dataset have numbers at center of the image.

## Model Summary
![alt image](https://user-images.githubusercontent.com/46129975/119969613-685b9c80-bfcc-11eb-8336-61512dddfb83.png)

## Fetching MNIST Dataset and Dataset Prep
```
train_set = datasets.MNIST('../data', 
                   train=True, 
                   download=True,
                   transform=transforms.Compose([
                                       transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) 
                                       # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       ]))

test_set = datasets.MNIST('../data', 
                   train=False, 
                   download=True,
                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

```
```
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)
```

# Defining Model Object, LR Scheduler and Optimizer

```
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
```

## Training Logs
We have used LR Scheduling.
```
EPOCH: 1 LR =  [0.01]
/usr/local/lib/python3.7/dist-packages/torch/optim/lr_scheduler.py:370: UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
  "please use `get_last_lr()`.", UserWarning)
Train set: Accuracy=87.5
Test set: Average loss: 0.0757, Accuracy: 9789/10000 (97.9%)

EPOCH: 2 LR =  [0.01]
Train set: Accuracy=97.8
Test set: Average loss: 0.0502, Accuracy: 9861/10000 (98.6%)

EPOCH: 3 LR =  [0.01]
Train set: Accuracy=98.3
Test set: Average loss: 0.0358, Accuracy: 9903/10000 (99.0%)

EPOCH: 4 LR =  [0.01]
Train set: Accuracy=98.5
Test set: Average loss: 0.0376, Accuracy: 9889/10000 (98.9%)

EPOCH: 5 LR =  [0.01]
Train set: Accuracy=98.7
Test set: Average loss: 0.0318, Accuracy: 9914/10000 (99.1%)

EPOCH: 6 LR =  [0.01]
Train set: Accuracy=98.8
Test set: Average loss: 0.0286, Accuracy: 9903/10000 (99.0%)

EPOCH: 7 LR =  [0.0001]
Train set: Accuracy=99.1
Test set: Average loss: 0.0219, Accuracy: 9931/10000 (99.3%)

EPOCH: 8 LR =  [0.001]
Train set: Accuracy=99.1
Test set: Average loss: 0.0218, Accuracy: 9932/10000 (99.3%)

EPOCH: 9 LR =  [0.001]
Train set: Accuracy=99.1
Test set: Average loss: 0.0210, Accuracy: 9936/10000 (99.4%)

EPOCH: 10 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0221, Accuracy: 9933/10000 (99.3%)

EPOCH: 11 LR =  [0.001]
Train set: Accuracy=99.1
Test set: Average loss: 0.0212, Accuracy: 9939/10000 (99.4%)

EPOCH: 12 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0210, Accuracy: 9936/10000 (99.4%)

EPOCH: 13 LR =  [1e-05]
Train set: Accuracy=99.2
Test set: Average loss: 0.0210, Accuracy: 9936/10000 (99.4%)

EPOCH: 14 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0207, Accuracy: 9939/10000 (99.4%)

EPOCH: 15 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0209, Accuracy: 9935/10000 (99.3%)

EPOCH: 16 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0208, Accuracy: 9939/10000 (99.4%)

EPOCH: 17 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0204, Accuracy: 9936/10000 (99.4%)

EPOCH: 18 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.4%)

EPOCH: 19 LR =  [1.0000000000000002e-06]
Train set: Accuracy=99.2
Test set: Average loss: 0.0205, Accuracy: 9939/10000 (99.4%)
```

## Plotting Graphs for Losses and Accuracies
```
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(training_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(training_accuracy)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(testing_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(testing_accuracy)
axs[1, 1].set_title("Test Accuracy")
```
![image](https://user-images.githubusercontent.com/46129975/120001384-59d3ac00-bff1-11eb-994c-93664b2d8975.png)

## Testing with Custom Images
Testing the model with custom handwritten images.
### Original Images
![image](https://user-images.githubusercontent.com/46129975/120000885-e92c8f80-bff0-11eb-9ad7-2aeab04ddd10.png)
### Images converted to MNIST format
![image](https://user-images.githubusercontent.com/46129975/120000981-f9dd0580-bff0-11eb-8067-0faba6c1a77f.png)
### Model Predictions
![image](https://user-images.githubusercontent.com/46129975/120001318-4b859000-bff1-11eb-8e6b-d2feaabe85b8.png)


