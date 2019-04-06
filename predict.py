import matplotlib.pyplot as plt

import torch
import numpy as np

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict

import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import copy
import argparse
#from train import load_model

parser = argparse.ArgumentParser(description='Process Command Line Input')
parser.add_argument('--image', type=str, help='Path to Image ')
parser.add_argument('--topk', type=int, default=5,help='Top Categories')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--checkpoint', type=str,default='checkpoint.pth', help='Checkpoint')
parser.add_argument('--category_names', type=str,default='cat_to_name.json', help='JSON File with Labels')
args = parser.parse_args()

image = args.image
topk = args.topk
checkpoint = args.checkpoint
category_names = args.category_names
gpu = args.gpu

def process_image(image):
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    from PIL import Image
    img = Image.open(image)
    image = image_transforms(img)
    return image.numpy()

display_image = process_image(image)
#print(display_image.shape)



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

#imshow(display_image)

def predict(image_path, checkpoint, topk=5, category_names=category_names,gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   
    # TODO: Implement the code to predict the class from an image file

    model = models.vgg19(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(.1)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
     
    model.classifier = classifier
    checkpoint_dict = torch.load(checkpoint)
    #arch = checkpoint_dict['arch']
    #arch = 'vgg19'
    #num_labels = len(checkpoint_dict['class_to_idx'])
    #hidden_units = checkpoint_dict['hidden_units']
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    #print(model.state_dict())
    #print(model.class_to_idx)
    model.eval()
    model.cpu()
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    labels = []
    for class_index in top_classes:
        labels.append(cat_to_name[str(class_index)])    
    #print('Top Classes: ', top_classes)
    print('Top Probs: ', top_prob.numpy()[0])
    print('Labels:',labels)
    return top_prob, top_classes

#print(model)    
#image_path = ('flowers/test' + '/1/' + 'image_06743.jpg')
#print(image_path)
#probs, classes = predict(image_path, model)

if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)
