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
parser.add_argument('flower_directory', type=str, action='store', default='flowers/test/10/image_07104.jpg',
                    help='Select location of flower to predict')
parser.add_argument('--load_classifier', dest='load_classifier', type=str, action='store', default='checkpoint.pth',
                    help='Load classifier')
parser.add_argument('--top_k', dest='top_k', action='store', default=5, type=int,
                    help='Select number of top predictions')
parser.add_argument('--category_names', action='store', dest='category_names', default='cat_to_name.json',
                    help='Input Category json file')
parser.add_argument('--gpu', dest='gpu', action='store_false', default=True,
                    help='Select CPU mode on or off. Default is GPU mode')
outputs = parser.parse_args()

flower_directory = outputs.flower_directory
load_classifier = outputs.load_classifier
top_k = outputs.top_k
category_names = outputs.category_names
gpu = outputs.gpu
#print('outputs', outputs)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Create variable
model = 'null'

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image_processed = img_transforms(img)
    return image_processed.numpy()


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Render model
    print('Loading model')
    model = load_checkpoint(model)
    print(model)
    # print(model)
    if gpu:
        model.to('cuda')
    else:
        pass
    # Render image
    img = process_image(image_path)
    img = torch.FloatTensor(img).unsqueeze(0)

    if gpu:
        img = img.to('cuda')
    else:
        pass

    model.eval()
    with torch.no_grad():
        output = model.forward(img)
    output = torch.exp(output)
    # print(output[0].sum())
    prob = np.array(output.topk(topk)[0])[0]
    labels = np.array(output.topk(topk)[1])[0]

    class_to_idx = model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes = []
    for index in labels:
        classes += [indx_to_class[index]]

    return prob, classes

probs, classes = predict(flower_directory, load_classifier, top_k)
# print(probs, classes)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

names = []
for i in classes:
    names += [cat_to_name[i]]

flower_predicted = names[0]
print('Flower predicted is: ', flower_predicted)
print('Probablity is', probs[0])
print('--------------------------')
print('Likelihood of these flowers in descending order')
for i,j in zip(names,probs):
    print('Name: ', i)
    print('Probability: ', round(j*100,5))
    print('-----------')
