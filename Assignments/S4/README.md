# Assignment 4 
# PART 1

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


- ***i1***, ***i2***                      -> ***inputs*** to the network
- ***w1***, ***w2***, ***w3***, ***w4***  -> ***weights*** from ***input*** to ***hidden layer 1***
- ***h1*** and ***h2***                   -> neuron in ***hidden layer 1***
- ***a_h1*** and ***a_h2***               -> output of ***activation functions*** applied to ***h1*** and ***h2*** respectively.
- ***w5***, ***w6***, ***w7***, ***w8***  -> ***weights*** from ***hidden layer 1*** to ***output layer***
- ***o1*** and ***o2***                   -> ***output*** neurons
- ***t1*** and ***t2***                   -> are ***target values/expected values***.
- ***a_o1*** and ***a_o2*** are output of ***activation functions*** applied to ***o1*** and ***o2*** respectively.
- ***E1*** and ***E2*** are ***error/loss*** of outputs a_o1 and a_o2 respectively.
- ***E_Total*** is ***sum*** of errors ***E1*** and ***E2***.


## Forward Pass Equations
The way input is propagated in forward pass is explained by below equations.
- The neurons in hidden layer 1 are weighted sum of input neurons.
    > h1 = w1*i1 + w2*i2
    > h2 = w3*i1 + w4*i2


