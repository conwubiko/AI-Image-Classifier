# */AIPND Image Classifier - Part2
# PROGRAMMER: Cyril O.
# DATE: CREATED: 26th October 2019
# PURPOSE: This part of the program uses the trained network to predicts the class for an input image.
# 
#
# To train a new network on a data set with train.py
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py /path/to/image checkpoint 
#       e.g. /Just simply type python predict.py
#             
#   Example call:
#    python predict.py
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
from math import floor


# Imports functions created fro this program
from get_input_args import get_input_args
from utils import load_checkpoint, load_cat_names

# Function to process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    new_size = [0, 0]
    
    if image.size[0] > image.size[1]:
        new_size = [image.size[0], 256]
    else:
        new_size = [256, image.size[1]]

    image.thumbnail(new_size, Image.ANTIALIAS)
    width, height = image.size  
    left = (256 - 224)/2
    upper = (256 - 224)/2
    right = (256 + 224)/2
    lower = (256 + 224)/2
    image = image.crop((left, upper, right, lower))
    image = np.array(image)/255.

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std
    
    # return a ndarray.transpose PIL image; reorder color channel first, 
    # and retain order of the other two dimensions
    image = np.transpose(image, (2, 0, 1)) 

    return image


def predict(image_path, model, topk, gpu):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Turn off training
    model.eval()
    
    # Use GPU otherwise use CPU
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    # Process the image using the function - process_image (as above)
    #image = process_image(image_path)
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    
    # Tranfer to tensor
    #image = torch.from_numpy(np.array([image])).float()
    
    if gpu and cuda:
        inputs = Variable(tensor.float().cuda())
    else:
        inputs = Variable(tensor)
    # The image becomes the input
    
    inputs = inputs.unsqueeze(0)
    logps = model.forward(inputs)
    
    # Calculate the prob function
    ps = torch.exp(logps).data.topk(topk)
    
    prob = ps[0].cpu()
    classes = ps[1].cpu()
    
    class_to_idx_mapped = {model.class_to_idx[i]: i for i in model.class_to_idx}
    mapped_classes = list()
    
    # transfer index to label
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_mapped[label])
    
    return prob.numpy()[0], mapped_classes


def main():
    in_arg = get_input_args()
    gpu = in_arg.gpu
    model = load_checkpoint(in_arg.checkpoint)
    
    cat_to_name = load_cat_names(in_arg.category_names)
    
    if in_arg.filepath == None:
        image_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(image_num) + '/'))
        img_path = './flowers/test/' + str(image_num) + '/' + image
        prob, classes = predict(img_path, model, in_arg.top_k, gpu)
        print("Selected Image is: " + str(cat_to_name[str(image_num)]))
    else:
        # Show random predicted image of the original image (displayed above) from a particular subfolder
        #image = random.choice(os.listdir('./flowers/test/' + str(image_num) + '/'))
        img_path = in_arg.filepath
        prob, classes = predict(img_path, model, in_arg.top_k, gpu)
        print("Selected Image is: " + img_path)

    print("\nProbabilities are: \n", prob)
    print("\nClasses are: \n", classes)
    print("\nFlowers names are: \n", [cat_to_name[i] for i in classes])

if __name__ == "__main__":
    main()