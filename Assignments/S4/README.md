# Assignment 4

## Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#table-of-contents)
- [Part-1 - Backpropagation](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#part-1---backpropagation)
    - [Excel File - Backpropagation](https://github.com/amanjain487/tsai-eva6/files/6562335/Session.-.4.-.BackPropagation.xlsx)
    - [Model Architecture](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#neural-network)
    - [Forward Pass Equations](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#forward-pass-equations)
    - [Backpropagation Equations](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#backpropagation-equations)
    - [Training Neural Network in Excel](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#training-neural-network-in-excel)
    - [Error Graph at Different Learning Rates](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#error-graph-at-different-learning-rates)
- [Part-2 - Architectural Basics](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#part-2---architectural-basics)
    - [Colab File](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/Session_4_Part_2.ipynb)
    - [Target](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#target)
    - [Model Architecture](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#model-architecture)
    - [Model Summary](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#model-summary)
    - [Fetching MNIST Dataset and Dataset Prep](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#fetching-mnist-dataset-and-dataset-prep)
    - [Defining Model Object, LR Scheduler and Optimizer](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#defining-model-object-lr-scheduler-and-optimizer)
    - [Training Logs](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#training-logs)
    - [Plotting Graphs for Losses and Accuracies](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#plotting-graphs-for-losses-and-accuracies)
    - [Testing with Custom Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#testing-with-custom-images)
        - [Original Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#original-images)
        - [Images converted to MNIST format](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#images-converted-to-mnist-format)
        - [Model Predictions](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#model-predictions) 

# PART-1 - BackPropagation
Derive the Backpropagation Equations and train a network in Excel.

## Neural Network

![alt image](https://cdn.mathpix.com/snip/images/85Qij7WDdP9k7huvKRKkOgHUhByuDWvhqVMy8gwp12w.original.fullsize.png)
[Link to Excel File](https://github.com/amanjain487/tsai-eva6/files/6562335/Session.-.4.-.BackPropagation.xlsx)
In the above neural network, 
|Variables|What are they?|From|To|
| :----: | :----: | :----: | :----: |
|i1, i2|Neurons|-|Input Layer|
|w1, w2, w3, w4|Weights|Input Layer|Hidden Layer 1|
|h1, h2|Neurons|Hidden Layer 1|-|
|a_h1, a_h2|Sigmoid Activated Neurons|Hidden Layer 1|Activated Hidden Layer|
|w5, w6, w7, w8|Weights|Hidden Layer 1|Output Layer|
|o1, o2|Neurons|Activated Output Layer|-|
|a_o1, a_o2|Sigmoid Activated Neurons|Output Layer|Activated Output Layer|
|t1, t2|Target Values|Expected Output|-|
|E1, E2|Error/Loss|Activated Output Layer|E_Total|
|E_Total|Total Error|Sum of E1, E2|-|

## Forward Pass Equations
The way input is propagated in forward pass is explained by below equations.
- The neurons in hidden layer 1 are weighted sum of input neurons.
    ```
    h1 = w1*i1 + w2*i2
    h2 = w3*i1 + w4*i2
    ```
- Sigmoid Activation Function is applied on ```h1``` and ```h2``` to make model non-linear.
    ```
    a_h1 = σ(h1) = 1/(1 + exp(-h1))
    a_h2 = σ(h2) = 1/(1 + exp(-h2))
    ```
- Neurons in output layer is the weighted sum of activated neurons in hidden layer 1.
    ```
    o1 = w5*a_h1 + w6*a_h2
    o2 = w7*a_h1 + w8*a_h2
    ```
- Sigmoid Activation Function is applied on ```o1``` and ```o2``` to make model non-linear.
    ```
    a_o1 = σ(o1) = 1/(1 + exp(-o1))
    a_o2 = σ(o2) = 1/(1 + exp(-o2))
    ```
- ```E1``` and ```E2``` are errors/loss of output ```o1``` and ```o2``` respectively.
    ```
    E1 = 1/2 * (t1 - a_o1)2	
    E2 = 1/2 * (t2 - a_o2)2	
    ```
- ```E_Total``` is the sum of errors/loss of 2 outputs.
    ```
    E_total = E1 + E2
    ```
    
## Backpropagation Equations
- We need to do perform backpropagation of loss/error to weights, so that we can update them for better predictions.
- The above achieved by differentiating total error with respective weights and update the existing weights with calculated gradients.
- So, to propagate the loss from total error to all weights, we will need partial derivative of each variable which is present in any possible path from weight to total error.

- Lets propagate error from total error to a_o1 and a_o2.
    ``` 
    ∂E_Total/∂a_o1 = ∂(E1+E2)/∂a_o1 = ((∂E1/∂a_o1) + (∂E2/∂a_o1))
        ∂E1/∂a_o1 = ∂(1/2 * (t1 - a_o1)2)/∂a_o1
        ∂E1/∂a_o1 = -(t1 - a_o1) = (a_o1 - t1)
        ∂E2/∂a_o1 = ∂(1/2 * (t2 - a_o2)2)/∂a_o1
        ∂E2/∂a_o1 = 0
    ∂E_Total/∂a_o1 = (a_o1 - t1)
    ```
    ``` 
    ∂E_Total/∂a_o2 = ∂(E1+E2)/∂a_o2 = ((∂E1/∂a_o2) + (∂E2/∂a_o2))
        ∂E1/∂a_o2 = ∂(1/2 * (t1 - a_o1)2)/∂a_o2
        ∂E1/∂a_o2 = 0
        ∂E2/∂a_o2 = ∂(1/2 * (t2 - a_o2)2)/∂a_o2
        ∂E2/∂a_o2 =  -(t2 - a_o2) = (a_o2 - t2)
    ∂E_Total/∂a_o2 = (a_o2 - t2)
    ```
- Backpropagate from a_o1 to o1 and a_o2 to o2.
    ```
    ∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)
    ```
    ```
    ∂a_o2/∂o2 = ∂(σ(o2))/∂o2 = a_o2 * (1 - a_o2)
    ```
- Backpropagate from o1, o2 to a_h1 and a_h2.
    ```
    ∂o1/∂a_h1 = ∂(w5*a_h1 + w6*a_h2)/∂a_h1 = w5
    ∂o2/∂a_h1 = ∂(w7*a_h1 + w8*a_h2)/∂a_h1 = w7
    ```
    ```
    ∂o1/∂a_h2 = ∂(w5*a_h1 + w6*a_h2)/∂a_h2 = w6
    ∂o2/∂a_h2 = ∂(w7*a_h1 + w8*a_h2)/∂a_h2 = w8
    ```
- Backpropagate from a_h1 to h1 and a_h2 to h2.
    ```
    ∂a_h1/∂h1 = ∂(σ(h1))/∂h1 = a_h1 * (1 - a_h1)
    ```
    ```
    ∂a_h2/∂h2 = ∂(σ(h2))/∂h2 = a_h2 * (1 - a_h2)
    ```
- Backpropagate from h1 to w1, w2 and h2 to w3, w4.
    ```
    ∂h1/∂w1 =∂( w1*i1 + w2*i2)/∂w1 = i1
    ∂h1/∂w2 =∂( w1*i1 + w2*i2)/∂w2 = i2
    ```
    ```
    ∂h2/∂w3 =∂( w3*i1 + w4*i2)/∂w3 = i1
    ∂h2/∂w4 =∂( w3*i1 + w4*i2)/∂w4 = i2
    ```
- Backpropagate from o1 to w5, w6 and o2 to w7, w8.
    ```
    ∂o1/∂w5 = ∂(w5*a_h1 + w6*a_h2)/∂w5 = a_h1
    ∂o1/∂w6 = ∂(w5*a_h1 + w6*a_h2)/∂w6 = a_h2
    ```
    ```
    ∂o2/∂w7 = ∂(w7*a_h1 + w8*a_h2)/∂w7 = a_h1
    ∂o2/∂w8 = ∂(w7*a_h1 + w8*a_h2)/∂w8 = a_h2
    ```
- Backpropagate from total error to a_h1 and and total error to a_h2
    ```
    ∂E_Total/∂a_h1 = ∂(E1+E2)/∂a_h1 = ((∂E1/∂a_h1) + (∂E2/∂a_h1))
        ∂E1/∂a_h1 = ∂E/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1
        ∂E1/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5
        ∂E2/∂a_h1 = ∂E/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h1
        ∂E2/∂a_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7
    ∂E_Total/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7
    ```
    ```
    ∂E_Total/∂a_h2 = ∂(E1+E2)/∂a_h2 = ((∂E1/∂a_h2) + (∂E2/∂a_h2))
        ∂E1/∂a_h2 = ∂E/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h2
        ∂E1/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6
        ∂E2/∂a_h2 = ∂E/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h2
        ∂E2/∂a_h2 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w8
    ∂E_Total/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8
    ```
- Total backpropagation from total error to weights w5, w6, w7 and w8
    ```
    ∂E_Total/∂w5 = ∂E_Total/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
    ∂E_Total/∂w5 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1
    ```
    ```
    ∂E_Total/∂w6 = ∂E_Total/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w6
    ∂E_Total/∂w6 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h2    
    ```
    ```
    ∂E_Total/∂w7 = ∂E_Total/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂w7
    ∂E_Total/∂w7 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h1
    ```
    ```
    ∂E_Total/∂w8 = ∂E_Total/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂w8
    ∂E_Total/∂w8 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h2
    ```
- Total backpropagation from total error to weights w1, w2, w3, w4
    ```
    ∂E_Total/∂w1 = ∂E_Total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
    ∂E_Total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
    ```
    ```
    ∂E_Total/∂w2 = ∂E_Total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
    ∂E_Total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
    ```
    ```
    ∂E_Total/∂w3 = ∂E_Total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
    ∂E_Total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
    ```
    ```
    ∂E_Total/∂w4 = ∂E_Total/∂a_h2 * ∂a_h2/∂h2 * ∂h1/∂w4
    ∂E_Total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
    ```
- The weights are updated using the below formula.
    ```
    ∀i|(1<=i<=8)|wi = wi - η*(∂(Total Error)/∂wi)
    ```
    where, η is learning rate(lr).
    

## Training Neural Network in Excel
- ### η = 0.1
[Link to Excel File](https://github.com/amanjain487/tsai-eva6/files/6562335/Session.-.4.-.BackPropagation.xlsx)
![alt image](https://cdn.mathpix.com/snip/images/7UFeoN8eTqDeUk8G5Ku_ke4pU6JjxbT29Egsmk7GK4A.original.fullsize.png)

## Error Graph at Different Learning Rates

![alt image](https://user-images.githubusercontent.com/46129975/119308205-6622d680-bc8a-11eb-90ff-ad7f67298da9.png)
- The model converges faster at larger learning rate, but when learning rate becomes too high, the model starts diverging.
- When the learning rate is too less, model takes lot of time/iteration to reach minima. At higher rates, sometimes model gets stuck at local minima.

# Part-2 - Architectural Basics
This part aims at designing a CNN model for MNIST Classification with below mentioned target.

## Target
[Link to Colab File](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/Session_4_Part_2.ipynb)

|Targets|Achieved?|
| :----: | :----: |
|99.4% validation accuracy|Yes - 99.45% in Epoch 9|
Less than 20k Parameters|Yes - 8320 Parameters used|
|Less than 20 Epochs|Yes - Model Consistent from 11th Epoch to 19th Epoch|
|Used Batch Normalization|Yes|
|Used Dropout|Yes|
|Used Fully Connected Layer|Yes|
|Used Global Averaging Pool Layer|Yes|

## Model Architecture
The model that is used to achieve the above targets is

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

## Defining Model Object, LR Scheduler and Optimizer

```
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
```

## Training Logs
We have used LR Scheduling.
```
EPOCH: 1 LR =  [0.01]
Train set: Accuracy=84.8
Test set: Average loss: 0.1158, Accuracy: 9659/10000 (96.6%)

EPOCH: 2 LR =  [0.01]
Train set: Accuracy=97.7
Test set: Average loss: 0.0483, Accuracy: 9866/10000 (98.7%)

EPOCH: 3 LR =  [0.01]
Train set: Accuracy=98.2
Test set: Average loss: 0.0380, Accuracy: 9885/10000 (98.8%)

EPOCH: 4 LR =  [0.01]
Train set: Accuracy=98.4
Test set: Average loss: 0.0363, Accuracy: 9882/10000 (98.8%)

EPOCH: 5 LR =  [0.01]
Train set: Accuracy=98.7
Test set: Average loss: 0.0270, Accuracy: 9916/10000 (99.2%)

EPOCH: 6 LR =  [0.01]
Train set: Accuracy=98.8
Test set: Average loss: 0.0318, Accuracy: 9906/10000 (99.1%)

EPOCH: 7 LR =  [0.0001]
Train set: Accuracy=99.0
Test set: Average loss: 0.0207, Accuracy: 9932/10000 (99.3%)

EPOCH: 8 LR =  [0.001]
Train set: Accuracy=99.1
Test set: Average loss: 0.0198, Accuracy: 9932/10000 (99.3%)

EPOCH: 9 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0190, Accuracy: 9945/10000 (99.5%)

EPOCH: 10 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0200, Accuracy: 9934/10000 (99.3%)

EPOCH: 11 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0195, Accuracy: 9937/10000 (99.4%)

EPOCH: 12 LR =  [0.001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0191, Accuracy: 9941/10000 (99.4%)

EPOCH: 13 LR =  [1e-05]
Train set: Accuracy=99.2
Test set: Average loss: 0.0194, Accuracy: 9939/10000 (99.4%)

EPOCH: 14 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99.4%)

EPOCH: 15 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0188, Accuracy: 9944/10000 (99.4%)

EPOCH: 16 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.4%)

EPOCH: 17 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0194, Accuracy: 9938/10000 (99.4%)

EPOCH: 18 LR =  [0.0001]
Train set: Accuracy=99.2
Test set: Average loss: 0.0189, Accuracy: 9939/10000 (99.4%)

EPOCH: 19 LR =  [1.0000000000000002e-06]
Train set: Accuracy=99.3
Test set: Average loss: 0.0188, Accuracy: 9944/10000 (99.4%)
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
![image](https://user-images.githubusercontent.com/46129975/120015873-81cb0b80-c001-11eb-9b56-0bed3492295b.png)

## Testing with Custom Images
Testing the model with custom handwritten images.
### Original Images
![image](https://user-images.githubusercontent.com/46129975/120015910-8e4f6400-c001-11eb-98cd-25c8a3d00708.png)
### Images converted to MNIST format
![image](https://user-images.githubusercontent.com/46129975/120015936-97403580-c001-11eb-9184-e35cc31f06cd.png)
### Model Predictions
![image](https://user-images.githubusercontent.com/46129975/120015955-9e674380-c001-11eb-88db-8c8f3495b568.png)

We tested the model for 4 handwritten images, and got correct output for 3 images, hence accuracy = ```100 * 3/4``` = ```75%``` 

The reason for less accuracy is:
- Not enough Handwritten images to test the model rigorously with handwritten images.
- These are handwritten images, and actual MNIST dataset on which model is trained follow certain properties like
    - Size of 28 x 28
    - Digit is at centre
    - Black background
    - Digit in white color
- Though, we try to convert them to the format followed by MNIST, we may not be successful in converting all images properly.
