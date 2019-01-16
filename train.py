import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
from PIL import Image
import json


parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type=str, action='store', default='flowers',
                    help='Select location for classifier training and testing')
parser.add_argument('--arch', dest='arch', action='store', type=str, default='vgg16',
                    help='Select alexnet or vgg16 which is default')
parser.add_argument('--learning_Rate', dest='learning_Rate', action='store', default=0.001, type=float,
                    help='Select learning rate')
parser.add_argument('--hidden_units', dest='hidden_units', action='store', default=500,  type=int,
                    help='Input no. of hidden units')
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=3,
                    help='Input number of epochs')
parser.add_argument('--gpu', dest='gpu', action='store_false', default=True,
                    help='Select CPU mode on or off. Default is GPU mode')
parser.add_argument('--save_dir', dest='save_directory', action='store',  default='checkpoint.pth')
outputs = parser.parse_args()

data_dir = outputs.data_directory
arch = outputs.arch
save_directory = outputs.save_directory
hidden_units = outputs.hidden_units
learning_Rate = outputs.learning_Rate
epochs = outputs.epochs
gpu = outputs.gpu
model = 'null'
input_features = 0

if arch == 'alexnet':
    print('Alexnet chosen as model')
    model = models.alexnet(pretrained=True)
    input_features = model.classifier[1].in_features
else:
    print('VGG16 chosen as model')
    model = models.vgg16(pretrained=True)
    input_features = model.classifier[0].in_features



train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Configuring our classifier


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_Rate)


# Putting the above into functions, so they can be used later

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device=gpu):
    epochs = epochs
    print_every = print_every
    steps = 0

    if device:
        model.to('cuda')
    else:
        pass

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            if device:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))

                running_loss = 0
                print('Checking accuracy using Validate dataset')
                check_validation_accuracy(validloader)


def check_accuracy(testloader):

    if gpu:
        model.to('cuda')
    else:
        pass
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / total))
    model.train()


def check_validation_accuracy(testloader):
    if gpu:
        model.to('cuda')
    else:
        pass
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = (criterion(outputs, labels) / total)

    print('Accuracy of the validation set: %d %%' % (100 * correct / total))
    print('Loss of the validation set is:', loss)
    model.train()

print('Training Model')
do_deep_learning(model, trainloader, 3, 40, criterion, optimizer, gpu)
print('Model trained')
print('Checking accuracy with test set')
check_accuracy(testloader)

model.class_to_idx = train_data.class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')
print('Model has been saved!')
