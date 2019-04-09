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
parser.add_argument('--input_units', default=25088,type=int, help='Number of input units')
parser.add_argument('--hidden_units', default=4096,type=int, help='Number of hidden units')
parser.add_argument('--gpu', action='store_true',default=False,help='Use GPU if available')
parser.add_argument('--epochs', type=int,default=5, help='Number of epochs')

args = parser.parse_args()

if args.arch:
    if args.arch == 'vgg19':
        # Load a pre-trained model
        arch = args.arch
        #model = models.vgg19(pretrained=True)
    elif args.arch=='alexnet':
        #model = models.alexnet(pretrained=True)
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
else:
        gpu = False
#print('gpu ',gpu)

if args.save_dir:
        checkpoint_path = args.save_dir

if args.input_units:
        input_units = args.input_units

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

if arch == 'vgg19':
            if torch.cuda.is_available:
                model = models.vgg19(pretrained=True)
            else:
                model = models.vgg19(pretrained=True,map_location='cpu')
elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
        #else:
            #model = models.alexnet(pretrained=True,map_location= lambda storage, loc : storage)
        #    model = models.alexnet(pretrained=True)
else:
            print("unrecognized architecture")

for param in model.parameters():
        param.requires_grad = False

    #from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(.1)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier
#print(model)

def load_model(checkpoint_path):

    if arch == 'vgg19':
            if torch.cuda.is_available:
                model = models.vgg19(pretrained=True)
            else:
                model = models.vgg19(pretrained=True,map_location='cpu')
    elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
        #else:
            #model = models.alexnet(pretrained=True,map_location= lambda storage, loc : storage)
        #    model = models.alexnet(pretrained=True)
    else:
            print("unrecognized architecture")

    for param in model.parameters():
        param.requires_grad = False

    #from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(.1)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier
    if torch.cuda.is_available():
        checkpoint_dict = torch.load(checkpoint_path)
    else:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    if torch.cuda.is_available():
       model.to('cuda')

    return model
#model = load_model('checkpoint.pth')
# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    optimizer.zero_grad()
    model.eval()
    for images, labels in testloader:
        if torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')
        #images.resize_(images.shape[0], 50176)

        output = model.forward(images)
        #output = output.to('cuda')
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy
#print("Loaded model from checkpoint")

#print(model)
#Now Training code
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#epochs = 5
print_every = 40
#print_every = 1
steps = 0
#print(epochs)
# change to cuda
#model.to('cuda')
model
#Choose based on GPU and CUDA Availability

if gpu and torch.cuda.is_available():
    model.to('cuda')
else:
    model.to('cpu')
#print("Checked whether GPU is set and cuda is available")
#epochs = 0
for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        if gpu and torch.cuda.is_available():
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        # Forward and backward passes

        outputs = model.forward(inputs)
        #print("Forward Pass")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("Optimized")
        running_loss += loss.item()

        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            #model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                #print("loss", test_loss)
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
        if gpu and torch.cuda.is_available():
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
checkpoint = {'input_units': input_units,
              'output_size': 102,
              'hidden_units': hidden_units,
              'arch': arch,
              'state_dict': model.state_dict(),
              'class_to_idx' :model.class_to_idx
             }

torch.save(checkpoint, checkpoint_path)


model = load_model(checkpoint_path)
#Load Model
