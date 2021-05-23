# Assignment 4 
# PART 1

## Neural Network

![alt image](https://cdn.mathpix.com/snip/images/85Qij7WDdP9k7huvKRKkOgHUhByuDWvhqVMy8gwp12w.original.fullsize.png)

In the above neural network, 
- ***i1*** and ***i2*** are ***inputs*** to the network
- ***w1***, ***w2***, ***w3***, ***w4*** are ***weights*** from ***input*** to ***hidden layer 1***
- ***h1*** and ***h2*** are neuron in ***hidden layer 1***
- ***a_h1*** and ***a_h2*** are output of ***activation functions*** applied to ***h1*** and ***h2*** respectively.
- ***w5***, ***w6***, ***w7***, ***w8*** are ***weights*** from ***hidden layer 1*** to ***output layer***
- ***o1*** and ***o2*** are ***output*** neurons
- ***t1*** and ***t2*** are ***target values/expected values***.
- ***a_o1*** and ***a_o2*** are output of ***activation functions*** applied to ***o1*** and ***o2*** respectively.
- ***E1*** and ***E2*** are ***error/loss*** of outputs a_o1 and a_o2 respectively.
- ***E_Total*** is ***sum*** of errors ***E1*** and ***E2***.


## Forward Pass Equations
The way input is propagated in forward pass is explained by below equations.
- The neurons in hidden layer 1 are weighted sum of input neurons.
```
***h1 = w1*i1 + w2*i2	
h2 = w3*i1 + w4*i2***
```

