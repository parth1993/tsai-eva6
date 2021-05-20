# Assignment 3 - NUMBER AND SUM PREDICTOR

It takes two inputs.
- Image
- Random Number

Finally, predicts the sum of number present in Image and Random Number.

# Table of contents

- [Project Title](#project-title)
- [Table of contents](#table-of-contents)
- [Explanation - Work Flow](#installation)
- [Data Representation](#usage)
- [Data Generation Strategy](#development)
- [How 2 Inputs are Combined?](#contribute)
- [How Results are Evaluated?](#license)
- [Our Results](#footer)
- [Which Loss Functions and Why?]
- [Footer]

# Explanation - Work Flow

- Create 2 Models
    - One for predicting the number present in Image Input
    - Second to perfrom and predict the addition
- ### Model 1 - MNIST Model
  - Use convolutional layers, relu activation function and max-pooling to build the model
  - Optimizer - Stochastic Gradient Descent (SGD randomly picks one data point from the whole data set to compute derivatives at each iteration to reduce the computations)
  - Loss Function - Negative log likelihood (It is used to minimize loss such that greater the confidence of correct class, lower the loss and lower the confidence of correct class, greater the loss)

![alt text](https://cdn.mathpix.com/snip/images/CpoT9c4tcXZYd_xIaByGDUSpFpK-RfyC8F24g6LH7rA.original.fullsize.png)

- ### Model 2 - Sum Model
  - Use fully connected layers only with 3 layers
  - Optimizer - Adam (For sparse gradients on noisy problems, since addition of 2 numbers is noisy because the 2 numbers can be any number)
  - Loss Function - Mean Squared Error (It is the best choice for Regression like problem)
![alt text](https://cdn.mathpix.com/snip/images/SFlkTFbriAthdkFeAxfMhZ-Xh1hdmV77E4cdFTRGWpI.original.fullsize.png)

- Train MNIST Model on MNIST Dataset
- Test Accuracy is around 98-99% for 1 to 5 epochs.
- Then generate the dataset for training Sum Model
- Load the dataset using Dataset class and train the sum model
- Both the models are trained individually.
- Finally, when an Image and Random Number is given,
  - Pass the image through MNIST model and predict the number
  - Pass the predicted number and random number in Sum model and predict the addition of those 2 numbers.
- So, the output of MNIST model along with random number is fed as input to Sum model







# Footer
[(Back to top)](#table-of-contents)

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->

Leave a star in GitHub, give a clap in Medium and share this guide if you found this helpful.

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
