# Assignment 4

# Group Members
- Harshita Sharma
- Pankaj S K
- Avichal Jain
- Aman Kumar

# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#table-of-contents)
- [Part-1 - Backpropagation](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#part-1---backpropagation)
    - [Model Architecture](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#neural-network)
    - [Forward Pass Equations](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README.md#forward-pass-equations)
    - [Backpropagation Equations](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#backpropagation-equations)
    - [Training Neural Network in Excel](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#training-neural-network-in-excel)
    - [Error Graph at Different Learning Rates](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#error-graph-at-different-learning-rates)
- [Part - 2](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#part-2)

# PART-1 - BackPropagation
Derive the Backpropagation Equations and train a network in Excel.

## Neural Network

![alt image](https://cdn.mathpix.com/snip/images/85Qij7WDdP9k7huvKRKkOgHUhByuDWvhqVMy8gwp12w.original.fullsize.png)

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
![alt image](https://cdn.mathpix.com/snip/images/7UFeoN8eTqDeUk8G5Ku_ke4pU6JjxbT29Egsmk7GK4A.original.fullsize.png)

## Error Graph at Different Learning Rates

![alt image](https://user-images.githubusercontent.com/46129975/119308205-6622d680-bc8a-11eb-90ff-ad7f67298da9.png)
- The model converges faster at larger learning rate, but when learning rate becomes too high, the model starts diverging.
- When the learning rate is too less, model takes lot of time/iteration to reach minima. At higher rates, sometimes model gets stuck at local minima.

# Part-2 - Architectural Basics
This part aims at designing a CNN model for MNIST Classification with below mentioned target.

## Target

|Targets|Achieved?|
| :----: | :----: |
|99.4% validation accuracy|Yes - 99.4% in in Epoch 9|
Less than 20k Parameters|Yes - 8320 Parameters used|
|Less than 20 Epochs|Yes - Model Consistent from 9th Epoch to 19th Epoch|
|Used Batch Normalization|Yes|
|Used Dropout|Yes|
|Used Fully Connected Layer|Yes|
|Used Global Averaging Pool Layer|Yes|

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
