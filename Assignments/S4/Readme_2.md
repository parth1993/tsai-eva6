# Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_2.md#group-members)
- [Architecture Diagram](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_2_2.md#table-of-contents)
    - [Part-2 - REWRITE COLAB CODE](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_2.md#part-1---backpropagation)
    - [Model Architecture](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_2.md#neural-network)
    - [Fetching Mnist Dataset and Dataset Prep](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S4/README_2.md#Fetching-Mnist-Dataset-and-Dataset-Prep)
    - [Defining the model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Defining-the-model)
    - [Defining model object, loss function and optimizer](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Defining-model-object,-loss-function-and-optimizer)
    - [Train and Test the Model](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Train-and-Test-the-Model)
    -[Plotting Graphs for Losses and Accuracies](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Plotting-Graphs-for-Losses-and-Accuracies)
    -[Custom Testing](https://github.com/amanjain487/tsai-eva6/tree/main/Assignments/S4#Custom-Testing)

# Group Members
- Harshita Sharma
- Pankaj S K
- Avichal Jain
- Aman Kumar

# Model Architecture
![2021-05-28](https://user-images.githubusercontent.com/16293041/119993770-90a5c400-bfe9-11eb-8afe-a164e5f7d34b.jpg)

# Fetching Mnist Dataset and Dataset Prep

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

# Defining the model

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # RF = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 28x28x1
        # output_size = 28x28x32
        # RF = 3


        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 10, 1),
            nn.Conv2d(10, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 28x28x32
        # output_size = 28x28x10
        # RF = 5

        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size = 28x28x10
        # output_size = 14x14x10
        # RF = 10


        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 14x14x10
        # output_size = 12x12x10
        # RF = 12

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        # input_size = 12x12x10
        # output_size = 10x10x10
        # RF = 14

        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 10x10x10
        # output_size = 8x8x10
        # RF = 16

        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 8x8x10
        # output_size = 6x6x16
        # RF = 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01),
            nn.ReLU()
        ) 
        # input_size = 6x6x16
        # output_size = 4x4x16
        # RF = 20

        self.gap1 = nn.AvgPool2d(4) 
        # input_size = 4x4x16
        # output_size = 1x1x16
        # RF = 20

        self.lin = nn.Sequential(
            torch.nn.Linear(16, 10),
        ) 
        # input_size = 1x1x16
        # output_size = 1x1x10
        # RF = 20

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap1(x)
        x = x.view(-1, 16)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
```

# Defining model object, loss function and optimizer

```
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
```


# Train and Test the Model
Using lr scheduling.
```
for epoch in range(1, 10):
    print("EPOCH:", epoch, "LR = ", scheduler.get_lr())
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
```

# Plotting Graphs for Losses and Accuracies
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

# Custom Testing


