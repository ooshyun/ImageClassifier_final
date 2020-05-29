# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from time import time, sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

from get_input_args_train import get_input_args

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# from workspace_utils import keep_awake, active_session
def train(data_path, model_name, learning_rate, hidden_units, epoch, device_gpu):
#   data_path = str(data_path)
  print("Value : data_path {} (type={}), model_name {} (type={}), learning_rate {} (type={}), hidden_units {} (type={}), epoch {}, (type={}), device {} (type={})".format(data_path, type(data_path), model_name, type(model_name), learning_rate, type(learning_rate), hidden_units, type(hidden_units), epoch, type(epoch), device_gpu, type(device_gpu)))
        
  train_dir = data_path + '/train'
  valid_dir = data_path + '/valid'
  test_dir = data_path + '/test'
  # TODO: Define your transforms for the training, validation, and testing sets
  data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ])

  test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                       ])

  # TODO: Load the datasets with ImageFolder
  image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
  test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
  valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)

  # TODO: Using the image datasets and the trainforms, define the dataloaders
  train_loader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, shuffle = True)
  test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
  valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)

  print("###########Loading Data##############")
  # TODO: Build and train your network
  if device_gpu == "gpu":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")
  print("###########Device is {}##############".format(device))
  print("###########GPU CHECK#################")
  model = models[model_name]
  #Froze the grad
  for param in model.parameters():
      param.required_grad = False

  #Build the classifier
  from collections import OrderedDict
  model.classifier = nn.Sequential(nn.Linear(25088, 2048),
                            nn.ReLU(),
                            nn.Dropout(p=0.25),
                            nn.Linear(2048, hidden_units),
                            nn.ReLU(),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))

  # TODO: Do validation on the test set
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = learning_rate)
  print("###########TRAIN START###############")
  model = model.to(device)
  epochs = int(epoch)
  training_losses = []
  validation_losses = []
  
  model.train()    
  for e in range(epochs):
      running_loss = 0
      for images, labels in train_loader:
          images = images.to(device) 
          labels = labels.to(device) 
          optimizer.zero_grad()          
          log_ps = model.forward(images)
          loss = criterion(log_ps, labels)
          loss.backward()          
          optimizer.step()
          running_loss += loss.item()          
      else:
          valid_loss = 0 
          accuracy = 0         
          with torch.no_grad():
              model.eval()
              for images, labels in valid_loader:
                  images = images.to(device) 
                  labels = labels.to(device)
                  logps = model.forward(images)
                  valid_loss +=  criterion(logps, labels)             
                  
                  ps = torch.exp(logps)
                  top_p, top_class = ps.topk(1, dim=1)
                  equals = top_class == labels.view(*top_class.shape)
                  accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                  
                  
          model.train()        
          training_losses.append(running_loss/len(train_loader))
          validation_losses.append(valid_loss/len(valid_loader))
          
          print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(training_losses[-1]),
                "Test Loss: {:.3f}.. ".format(validation_losses[-1]),
                "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

  # TODO: Save the checkpoint 
  filename_pth = 'checkpoint_flower.pth'
  checkpoint = {
      'model_type': models[model_name],
      'model' : model,
      'learning_rate' : learning_rate,
      'batch_size' : 64,
      'epoch' : epochs,
      'classifier': model.classifier,
      'optimizer_dict': optimizer.state_dict(),
      'state_dict': model.state_dict(),
      'class_to_idx': image_datasets.class_to_idx
  }
  torch.save(checkpoint, filename_pth)
  # torch.save(model.state_dict, optimizer.state_dict, filename_pth)


def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    in_arg = get_input_args()

    ################
    train(in_arg.data_directory, in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.epochs, in_arg.gpu)
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
