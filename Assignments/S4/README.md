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

# Part-2

