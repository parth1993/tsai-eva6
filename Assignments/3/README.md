# Assignment 3 - DIGIT RECOGNIZER AND SUM PREDICTOR

The project takes two inputs.
- Image
- Random Number

Finally, predicts the number present in image and sum of number present in Image and Random Number.

# Table of contents

- [Project Title](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#assignment-3---digit-recognizer-and-sum-predictor)
- [Table of contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#table-of-contents)
- [Code File](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/Digit_Recognizer_and_Sum_Predictor.ipynb)
- [Explanation - Work Flow](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#explanation---work-flow)
    - [MNIST Model](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#model-1---mnist-model)
    - [Sum Model](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#model-2---sum-model)
- [Data Representation](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#data-representation)
- [Data Generation Strategy](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#data-generation-strategy)
- [How 2 Inputs are Combined?](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#how-2-inputs-are-combined)
- [Results and Evaluation](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#results-and-evaluation)
- [Which Loss Functions and Why?](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#which-loss-functions-and-why)
- [Training Shots](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#training-shots)
- [Footer](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#footer)
    - [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#group-6)

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
- Test Accuracy is around 99% in 5 epochs.
- Then generate the dataset for training Sum Model
- Load the dataset using Dataset class and train the sum model
- Both the models are trained individually.
- Finally, when an Image and Random Number is given,
  - Pass the image through MNIST model and predict the number
  - Pass the predicted number and random number in Sum model and predict the addition of those 2 numbers.
- So, the output of MNIST model along with random number is fed as input to Sum model

# Data Representation
- Input Image
    - Grayscale Image - So number of input channel = 1
    - Image Size = 28x28
    - Each image was represented as tensor of shape (number_of_images, number_of_channels, number_of_rows, number_of_columns) i.e., (1,1,28,28)
    - When the above Image was fed to model, it gave tensor of shape (1,1,1,10) as output,
        - where 10 is for number of classes (0-9)
    - Finally, the index of largest value(a scalar) was extracted and that is the prediction given by model for number present in image.
- Input Random Number
    - In Training,
        - Random pairs and their sum were generated.
        - Input for Sum Model was tensor of shape (1,2)
        - Input was converted to FloatTensor
        - Output for Sum Model was a scalar.
    - In Testing,
        - Output of MNIST Model and Random Number were combined as single tensor of shape (1,2)
        - Output of Sum Model was again a scalar.
        
# Data Generation Strategy

- MNIST Image Dataset
    - It is already available and is downloaded and used using Pytorch library.

- Random Numbers and its Sum
    - A function was written, which will generate 2 random numbers as input and add those 2 for output, thus creating a training instance.
 
- Finally, when model was evaluated, only a image from dataset and random number was given, the other number required was generated by MNIST model which predicts number present in input image.

# How 2 Inputs are Combined?

![alt text](https://cdn.mathpix.com/snip/images/5lXcVYKAa-DVmmVWRArtvDpRsgmhi1bvrE_IoOx3O54.original.fullsize.png)

# Results and Evaluation

- The results of MNIST model which predicts number present in Image were evaluated and its accuracy is around 99% 5 epochs.
- Accuracy kept increasing with each epoch.
- The results of Sum model was evaluated using RMSE(Root Mean Square Error) and its accuracy is around 25% to 30%.
![alt text](https://cdn.mathpix.com/snip/images/YNAiTD4c-62SRLgZs7BmcilFjuXAB-AGBel8o50o4Pg.original.fullsize.png)
![alt text](https://cdn.mathpix.com/snip/images/HtoW5hGL_qhqgmb0MFps1Q6IvRg_SY48d6IkkaTBoQo.original.fullsize.png)
![alt text](https://cdn.mathpix.com/snip/images/gUj3ooDnAn2XqvCG1309bQIZ0OixTIaFrqzAksU0ptA.original.fullsize.png)

# Which Loss Functions and Why?
- ### MNIST Model
    - Negative log likelihood
        - It is used to minimize loss such that greater the confidence of correct class, lower the loss and lower the confidence of correct class, greater the loss
        - Since, the task of MNIST Model is classification between 10 classes, we want our model to predict the correct class with high accuracy.
        - So, we chose NLL Loss Function to train model such that it learns to predict with high confidence.
- ### Sum Model
    - Mean Squared Error
        - This is the best loss function for Regression like problem.
        - The sum of two numbers also looks like regression problem.

# Training Shots
- ### MNIST Model - 5 epoch
![alt text](https://cdn.mathpix.com/snip/images/N8XIEjUlauvfIspl21NTYkWxAMG3Eh9Dj8W2foU1Lew.original.fullsize.png)
- ### Sum Model - 50 epoch
![alt text](https://cdn.mathpix.com/snip/images/7FezQk6CKv8YQLDr36EWqbvH7dx-veBIR0SD4DN2Ypg.original.fullsize.png)

# Footer
[(Back to table of contents)](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/3/README.md#table-of-contents)

### Group 6
    - Harshita Sharma
    - Pankaj S K
    - Avichal Jain
    - Aman Kumar
