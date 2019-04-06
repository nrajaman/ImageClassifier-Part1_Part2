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

parser = argparse.ArgumentParser(description='Process Command Line Input')
parser.add_argument('--data_dir', type=str, default='flowers',help='Path to source data ')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth',help='Save Checkpoint for Trained Model')
parser.add_argument('--arch', type=str, default='vgg19',help='Model architecture')
parser.add_argument('--learning_rate', default=.001,type=float, help='Learning rate')
parser.add_argument('--hidden_units', default=4096,type=int, help='Number of hidden units')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int,default=5, help='Number of epochs')

args = parser.parse_args()

if args.arch:
    if args.arch == 'vgg19':
        # Load a pre-trained model
        arch = args.arch
        model = models.vgg19(pretrained=True)
    elif args.arch=='alexnet':
        model = models.alexnet(pretrained=True)
        arch = args.arch
    else:
        raise ValueError('can support only vgg19 and alexnet - unsupported architecture ', arch)
        
if args.hidden_units:
        hidden_units = args.hidden_units

if args.epochs:
        epochs = args.epochs
            
if args.learning_rate:
        learning_rate = args.learning_rate

if args.gpu:
        gpu = args.gpu

if args.save_dir:
        checkpoint = args.save_dir
        
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

train_data = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(args.data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model():
   model = models.vgg19(pretrained=True)
   for param in model.parameters():
     param.requires_grad = False
 
   from collections import OrderedDict
   classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
  
    
   model.classifier = classifier
   return model
model = load_model()
# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    optimizer.zero_grad()
    model.eval()
    for images, labels in testloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        #images.resize_(images.shape[0], 50176)

        output = model.forward(images)
        #output = output.to('cuda')
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#epochs = 5
print_every = 40
steps = 0

# change to cuda
#model.to('cuda')
model
model.to('cuda')
for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        
        outputs = model.forward(inputs)
       
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            #model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                #test_loss = test_loss.to('cuda')
                #accuracy = accuracy.to('cuda')
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validate Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Validate Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            running_loss = 0
            # Make sure training is back on
        model.train()
        
# TODO: Do validation on the test set
correct = 0
total = 0
#model.to('cpu')
with torch.no_grad():
    model.eval()
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        #print(images.type, images.shape, torch.typename,images[0])
        outputs = model(images)
        #print("Processing")
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


#Save Checkpoint

model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_units': hidden_units,
              'arch': arch,
              'state_dict': model.state_dict(),
              'class_to_idx' :model.class_to_idx
             }

torch.save(checkpoint, 'checkpoint.pth')

#Load Model
def load_model(checkpoint_path):
    model = models.vgg19(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(.1)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
     
    model.classifier = classifier
    checkpoint_dict = torch.load(checkpoint_path)
    
    #arch = 'vgg19'
    #num_labels = len(checkpoint_dict['class_to_idx'])
    #hidden_units = checkpoint_dict['hidden_units']
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    #model.arch = checkpoint_dict['arch']
    #checkpoint = torch.load(checkpoint_path)
    #model = models.densenet121(pretrained=True)
    #model.load_state_dict(checkpoint['state_dict'])
    #class_to_idx = checkpoint['class_to_idx']
    #classifier = nn.Sequential(OrderedDict([
    #                      ('fc1', nn.Linear(1024, 500)),
    #                      ('relu', nn.ReLU()),
    #                      ('drop',nn.Dropout(.5)),
    #                      ('fc2', nn.Linear(500, 102)),
    #                      ('output', nn.LogSoftmax(dim=1))
    #                      ]))
  
    
    #model.classifier = classifier
    return model
model = load_model('checkpoint.pth')
#print(model)