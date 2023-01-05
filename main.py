import argparse
import sys

import torch
from torch import nn, optim
import click
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    trainloader = DataLoader(train_set, batch_size=16)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5
    steps = 0

    train_loss, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1).float()
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        else:
            train_loss.append(float(running_loss)/len(trainloader))
            print(f"Training loss: {running_loss/len(trainloader)}")

    #plot train loss
    fig = plt.figure(figsize=(12,4))
    # plt.subplot(1, 2, 1)
    plt.plot(range(1,1+epochs), train_loss, label='train_loss')
    plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(range(epochs)+1, train_accs, label='train_accs')
    # plt.legend()
    plt.show()
    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    model.eval()

    testloader = DataLoader(test_set, batch_size=16)
    equals = torch.empty(0)
    with torch.no_grad():
        # validation pass here
        for images, labels in testloader:
            images = images.view(images.shape[0], -1).float()
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = torch.cat((equals,(top_class == labels.view(*top_class.shape))))
                
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    