import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time, sleep
import torch, random, os
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import json
from get_input_args_predict import get_input_args
from collections import OrderedDict

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

#models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.parameters(), lr=0.004)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
   
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    print("#####Load Complete#####")
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
    # TODO: Process a PIL image for use in a PyTorch model
    # TODO: Define your transforms for the training, validation, and testing sets
    #Resizing
    width, height = image.size
    size = image.size
    image = image.resize(size)
    
    #Center crop
    new_width, new_height = image.size
    desired_size = 224, 224
    width_diff, height_diff = round(desired_size[0]), round(desired_size[1])
    left = round(new_width - width_diff)/2
    top = round(new_height - height_diff)/2
    right = round(new_width + width_diff)/2
    bottom = round(new_height + height_diff)/2
    
    image1 = image.crop((left, top, right, bottom))

    #Normalization
    image2 = np.array(image1) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image = (image2 - mean)/std
    image_final = np_image.transpose((2,0,1))

    return image_final


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_k, gpu_usage):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    ImageFolder = os.listdir(image_path)
    Random_num_dir = random.randint(0, len(ImageFolder) - 1)
    ImageList = os.listdir(image_path + "/" + ImageFolder[Random_num_dir])
    loadedImages = []
    for image in ImageList:
        if image[-3:] in ["png", "jpg"]:
            img = Image.open(image_path + "/" + ImageFolder[Random_num_dir] + "/" +image)
            loadedImages.append(img)
    Random_num = random.randint(0, len(loadedImages)) -1
    image_pc = process_image(loadedImages[Random_num])
    image_show = torch.from_numpy(image_pc)

    image_torch = torch.from_numpy(image_pc).type(torch.FloatTensor)
    image_unsq = image_torch.unsqueeze_(0)
    
    if gpu_usage == "gpu":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device("cpu")
    model.to(device)
    image_model = image_unsq.to(device)

    logps = model.forward(image_model)
    ps = torch.exp(logps)
    top_probes, top_classes = ps.topk(top_k, dim=1)

    return top_probes, top_classes, image_show
    # TODO: Implement the code to predict the class from an image file

def classifier(image_path, checkpoint, cat_to_name, top_k, gpu_usage): 

#     print(cat_to_name)    
    print("#####CAT NAME COMPLETE#####")
    filename_pth = checkpoint
    model, optimizer = load_checkpoint(filename_pth)
    print(model, optimizer)
    print("#####MODEL LOADING COMPLETE#####")
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    # # TODO: Display an image along with the top 5 classes

    # TODO: Display an image along with the top 5 classes
    probs, classes, image = predict(image_path, model, top_k, gpu_usage)
    print("#####PREDICT COMMPLETE#####")
    probs_cpu = probs.to(torch.device("cpu"))
#     print(probs_cpu.numpy(), classes, type(probs), type(classes))
    probs_cpu = probs_cpu.detach().numpy()
    indexes = []
    for index in classes[0]:
        indexes.append(str(int((index))))
    nums = []
    for index in probs[0]:
        nums.append(float((index)))
    i=0
    for index in nums:
        if index == probs.max():
            max_index = indexes[i]
        i += 1    

    names = []
    for index in indexes:
        names.append(cat_to_name[index])

    #Plot
    y_pos = np.arange(len(names))
    performance = np.array(probs_cpu)[0]

    print("name ",format(names))
    print("probability : ",format(performance))


def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    in_arg = get_input_args()
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    ################
    classifier(in_arg.test_image_path, in_arg.checkpoint, cat_to_name, in_arg.top_k, in_arg.gpu_usage)
    ################

    # TODO 0: Measure total program runtime by collecting end time
    end_time = time()
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
