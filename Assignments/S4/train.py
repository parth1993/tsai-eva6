import datetime
from os.path import join

import click
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from absl import logging
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm

from model import Net
from torchsummary import summary
from torchvision import datasets, transforms
from utils.general import create_save_dir, load_yaml


@click.group()
@click.option("--seed", default=1)
@click.pass_context
def train(ctx, seed):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(-1)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    ctx.ensure_object(dict)
    ctx.obj["use_cuda"] = use_cuda


def _train_model(model, train_loader, optimizer, device):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    training_loss = 0
    training_accuracy = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        training_loss = loss
        loss.backward()
        optimizer.step()
        predictions = output.argmax(dim=1, keepdim=True)
        correct += predictions.eq(target.view_as(predictions)).sum().item()
        processed += len(data)
        pbar.set_description(desc=f"Train set: Accuracy={100*correct/processed:0.1f}")
        training_accuracy = 100 * correct / processed
        return training_loss, training_accuracy


def _test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    testing_loss = 0
    testing_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()

    testing_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    testing_accuracy = 100.0 * correct / len(test_loader.dataset)
    return testing_loss, testing_accuracy


@train.command()
@click.argument("config")
@click.pass_context
def train_model(ctx, config):
    config = load_yaml(config).get("train")
    use_cuda = ctx.obj["use_cuda"]

    tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_dir = config["model_save_dir"]
    model_save_dir = join(model_save_dir, tag)
    create_save_dir(model_save_dir)

    train_set = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
            ]
        ),
    )

    test_set = datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    # torch.manual_seed(1)
    # if use_cuda:
    #     torch.cuda.manual_seed(1)

    batch_size = config["batch_size"]

    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, **kwargs
    )

    logging.info("Training  model !!!")
    model = Net()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    print(summary(model, input_size=(1, 28, 28)))

    if use_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    epochs = config["epochs"]

    # Tracking loss and accuracy for plotting graphs
    training_losses = []
    testing_losses = []
    training_accuracy = []
    testing_accuracy = []

    for epoch in range(1, epochs + 1):
        print("---------EPOCH: ", epoch," ------ LR: ", scheduler.get_lr())
        tr_loss, tr_acc = _train_model(model, train_loader, optimizer, device=device)

        training_losses.append(tr_loss)
        training_accuracy.append(tr_acc)

        te_loss, te_acc = _test_model(model, test_loader, device=device)
        testing_losses.append(te_loss)
        testing_accuracy.append(te_acc)

        scheduler.step()

    # Plotting Graphs for Losses and Accuracies
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(training_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(training_accuracy)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(testing_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(testing_accuracy)
    axs[1, 1].set_title("Test Accuracy")


if __name__ == "__main__":
    train()
