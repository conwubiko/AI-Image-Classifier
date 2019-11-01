
# */AIPND Image Classifier - Part2
# PROGRAMMER: Cyril O.
# DATE: CREATED: 26th October 2019
# PURPOSE: Now that we've built and trained a deep neural network on the flower data set, 
#          it's time to convert it into an application that others can use. 
#          Our application will use a pair of Python scripts that run from the command line. 
#          For testing, we'll use the checkpoint you saved in the first part.
# 
#
# To train a new network on a data set with train.py
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory with train images> --arch <model>
#             
#   Example call:
#    python train.py --arch vgg19
##

# Imports python modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import os, random
from time import time


# Imports functions created fro this program
from get_input_args import get_input_args
from utils import save_checkpoint, load_checkpoint
# from save_model import save_model

def train_classifier(model, criterion, optimizer, dataloaders, epochs, gpu):
    
    # Use GPU if it's available
    cuda = torch.cuda.is_available() 
    
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    
    running_loss = 0
    accuracy = 0
    
    start_time = time()
    print("\n========Network Training starts=======: ", start_time)
    for epoch in range(epochs):           
        for images, labels in dataloaders[0]:
            model.train(True)
            
            # move the images to GPU
            images, labels = images.cuda(), labels.cuda()
            
            # Zero our gradients
            optimizer.zero_grad()
            
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        else:
            model.eval()
            validation_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                for images, labels in dataloaders[1]:
                    # move the images to GPU
                    images, labels = images.cuda(), labels.cuda()
                    
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                    
                    #calculate our accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders[0]):.4f} "
              f"Validation loss: {validation_loss/len(dataloaders[1]):.4f} "
              f"Validation accuracy: {accuracy/len(dataloaders[1])*100:.2f}%")
        running_loss = 0

    
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    print("Starting training .....:", start_time)
    in_arg = get_input_args()

    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir, transform = train_transform),
                      datasets.ImageFolder(valid_dir, transform = valid_transform),
                      datasets.ImageFolder(test_dir, transform = test_transform)]


    # TODO: Using the image datasets and the transforms to specify the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                  torch.utils.data.DataLoader(image_datasets[1], batch_size=32), 
                  torch.utils.data.DataLoader(image_datasets[2], batch_size=32)]
    
    # model = getattr(models, in_arg.arch)(pretrained=True)
    # Loading a vgg19 pretrained model
    model = models.vgg19(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    if in_arg.arch == "vgg13":
        input_feature = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(input_feature, 1536),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(1536, 102),
                                   nn.LogSoftmax(dim=1))
    elif in_arg.arch == "vgg19":
        # Define a new, untrained feed-forward network to use as a classifier using the features as input
        # using ReLU activations and dropout
        input_feature = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(input_feature, 1536),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(1536, 102),
                                   nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    
    # Define Hyperparameters
    learn_rate = in_arg.learning_rate
    epochs = in_arg.epochs
        
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    class_index = image_datasets[0].class_to_idx
    gpu = in_arg.gpu
    train_classifier(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, in_arg, classifier)

    
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    
    

# Call to main function to run the program
if __name__ == "__main__":
    main()


