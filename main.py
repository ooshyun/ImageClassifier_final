# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

from workspace_utils import keep_awake, active_session

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# # TODO: Define your transforms for the training, validation, and testing sets
# data_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                       transforms.RandomCrop(224),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])
#                                      ])
#
# test_transforms = transforms.Compose([transforms.Resize(255),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])
#                                      ])
#
# # TODO: Load the datasets with ImageFolder
# image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
# test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
# valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
#
# # TODO: Using the image datasets and the trainforms, define the dataloaders
# train_loader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
# valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
#
# # import json
#
# # with open('cat_to_name.json', 'r') as f:
# #     cat_to_name = json.load(f)
#
# # print(cat_to_name)
#
# # TODO: Build and train your network
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #model = models.resnet34(pretrained=True)
# model = models.vgg16(pretrained=True)
# #model = models.alexnet(pretrained=True)
# # model
#
# # dataiter = iter(train_loader)
# # print(len(train_loader))
# # images, labels = dataiter.next()
# # print(images.shape)
# # print(images.view(images.shape[0], -1))
#
# #Froze the grad
# for param in model.parameters():
#     param.required_grad = False
#
# #Build the classifier
# from collections import OrderedDict
# model.classifier = nn.Sequential(nn.Linear(25088, 2048),
#                           nn.ReLU(),
#                           nn.Dropout(p=0.25),
#                           nn.Linear(2048, 512),
#                           nn.ReLU(),
#                           nn.Linear(512, 102),
#                           nn.LogSoftmax(dim=1))
# #                         (nn.Linear(25088, 512),
# #                           nn.ReLU(),
# #                           nn.Dropout(0.2),
# #                           nn.Linear(10000, 5000),
# #                           nn.ReLU(),
# #                           nn.Dropout(0.2),
# #                           nn.Linear(5000, 500),
# #                           nn.ReLU(),
# #                           nn.Dropout(0.2),
# #                           nn.Linear(512, 102),
# #                           nn.LogSoftmax(dim=1))
#
# # TODO: Do validation on the test set
# criterion = nn.CrossEntropyLoss()
# # nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.004)
# # optim.Adam(model.parameters(), lr = 0.001)
# model = model.to(device)
# epochs = 10
# training_losses = []
# validation_losses = []
# model.train()
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in train_loader:
#
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # print(images.shape)
#
#         optimizer.zero_grad()
#
#         log_ps = model.forward(images)
#         loss = criterion(log_ps, labels)
#         loss.backward()
#
#         optimizer.step()
#
#         # print(loss.item())
#         running_loss += loss.item()
#
#     else:
#         valid_loss = 0
#         accuracy = 0
#
#         with torch.no_grad():
#             model.eval()
#             for images, labels in valid_loader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 logps = model.forward(images)
#                 valid_loss += criterion(logps, labels)
#
#                 ps = torch.exp(logps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#
#         model.train()
#         training_losses.append(running_loss / len(train_loader))
#         validation_losses.append(valid_loss / len(valid_loader))
#
#         print("Epoch: {}/{}.. ".format(e + 1, epochs),
#               "Training Loss: {:.3f}.. ".format(training_losses[-1]),
#               "Test Loss: {:.3f}.. ".format(validation_losses[-1]),
#               "Test Accuracy: {:.3f}".format(accuracy / len(valid_loader)))
#
#         plt.plot(training_losses, label='Training loss')
#         plt.plot(validation_losses, label='Validation loss')
#         # plt.plot(Accuracy_list, label = 'Accuracy')
#         plt.legend(frameon=False)
#
#         # # TODO: Save the checkpoint
#         # filename_pth = 'vgg16_flower.pth'
#         # checkpoint = {
#         #     'optimizer': optim.SGD(model.parameters(), lr=0.004),
#         #     'arch': 'vgg16',
#         #     'classifier': model.classifier,
#         #     'state_dict': model.state_dict(),
#         #     'optimizer_dict': optimizer.state_dict(),
#         #     'class_to_idx': image_datasets.class_to_idx
#         # }
#         # torch.save(checkpoint, filename_pth)
#         # # torch.save(model.state_dict, optimizer.state_dict, filename_pth)
#
# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = models.vgg16(pretrained=True)
#     model.arch = checkpoint['arch']
#     model.class_to_idx = checkpoint['class_to_idx']
#     model.classifier = checkpoint['classifier']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_dict'])
#
#     for param in model.parameters():
#         param.requires_grad = False
#
#     return model
#
#
# filename_pth = 'vgg16_flower.pth'
# optimizer = optim.SGD(model.parameters(), lr=0.004)
#
# # TODO: Write a function that loads a checkpoint and rebuilds the model
# model = load_checkpoint(filename_pth)
# print(model)
#
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# # > [0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# # > ['70', '3', '45', '62', '55']
#
# import glob, os, random
#
# # TODO: Process a PIL image for use in a PyTorch model
# # TODO: Define your transforms for the training, validation, and testing sets
# image_path = test_dir
# ImageFolder = os.listdir(image_path)
# Random_num = random.randint(0, len(ImageFolder) - 1)
# ImageList = os.listdir(image_path + "/" + ImageFolder[Random_num])
# loadedImages = []
#
# for image in ImageList:
#     if image[-3:] in ["png", "jpg"]:
#         img = Image.open(image_path + "/" + ImageFolder[Random_num] + "/" + image)
#         loadedImages.append(img)
#
# random_num = random.randint(0, len(loadedImages) - 1)
# image = loadedImages[1]
#
# # Resizing
# width, height = image.size
# imagex = int(image.size[0])
# imagey = int(image.size[1])
# size = 255, 255
# # if imagex > imagey:
# #     image.thumbnail((100000, 256))
# # else:
# #     image.thumbnail((256, 100000))
#
# # if width < height:
# #     portait = True
# #     size = 255, round(height / (width / 255))
# # else:
# #     portrait = False
# #     size = round(width / (height / 255)), 255
# image = image.resize(size)
#
# # Center crop
# print(image.size)
# new_width, new_height = image.size
# desired_size = 224, 224
# width_diff, height_diff = round(desired_size[0]), round(desired_size[1])
# print(width_diff, height_diff)
# left = (new_width - width_diff) / 2
# top = (new_height - height_diff) / 2
# right = (new_width + width_diff) / 2
# bottom = (new_height + height_diff) / 2
#
# image1 = image.crop((left, top, right, bottom))
#
# # Normalization
# # print(image1)
# image2 = np.array(image1) / 255
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# np_image = (image2 - mean) / std
# image_final = np_image.transpose((2, 0, 1))
# print(image_final)
# image_ts = torch.tensor(np.array(image_final, dtype=float))
#
# print(image_ts)
#
# import glob, os
#
#
# def process_image(image):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
#     # TODO: Process a PIL image for use in a PyTorch model
#     # TODO: Define your transforms for the training, validation, and testing sets
#     #     image_path = test_dir
#     #     ImageFolder = os.listdir(image_path)
#     #     Random_num = random.randint(0, len(ImageFolder) - 1)
#     #     ImageList = os.listdir(image_path + "/" + ImageFolder[Random_num])
#     #     loadedImages = []
#     #     for image in ImageList:
#     #         if image[-3:] in ["png", "jpg"]:
#     #             img = Image.open(image_path + "/" + ImageFolder[Random_num] + "/" +image)
#     #             loadedImages.append(img)
#
#     #     random_num = random.randint(0, len(loadedImages) -1)
#     #     image = loadedImages[1]
#
#     # Resizing
#     width, height = image.size
#     size = image.size
#     #     imagex = int(image.size[0])
#     #     imagey = int(image.size[1])
#     #     if imagex > imagey:
#     #         image.thumbnail((100000, 256))
#     #     else:
#     #         image.thumbnail((256, 100000))
#
#     #     if width < height:
#     #         portait = True
#     #         size = 255, round(height / (width / 255))
#     #     else:
#     #         portrait = False
#     #         size = round(width / (height / 255)), 255
#     image = image.resize(size)
#
#     # Center crop
#     new_width, new_height = image.size
#     desired_size = 224, 224
#     width_diff, height_diff = round(desired_size[0]), round(desired_size[1])
#     left = round(new_width - width_diff) / 2
#     top = round(new_height - height_diff) / 2
#     right = round(new_width + width_diff) / 2
#     bottom = round(new_height + height_diff) / 2
#
#     image1 = image.crop((left, top, right, bottom))
#
#     # Normalization
#     image2 = np.array(image1) / 255
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     np_image = (image2 - mean) / std
#     image_final = np_image.transpose((2, 0, 1))
#
#     return image_final
#
#
# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
#
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#
#     return ax
#
#
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# # > [0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# # > ['70', '3', '45', '62', '55']
#
# import random
# import os
#
#
# def predict(image_path, model, topk=5):
#     ''' Predict the class (or classes) of an image using a trained deep learning model.
#     '''
#     ImageFolder = os.listdir(image_path)
#     Random_num_dir = random.randint(0, len(ImageFolder) - 1)
#     ImageList = os.listdir(image_path + "/" + ImageFolder[Random_num_dir])
#     loadedImages = []
#     for image in ImageList:
#         if image[-3:] in ["png", "jpg"]:
#             img = Image.open(image_path + "/" + ImageFolder[Random_num_dir] + "/" + image)
#             loadedImages.append(img)
#     Random_num = random.randint(0, len(loadedImages)) - 1
#     image_pc = process_image(loadedImages[Random_num])
#     image_show = torch.from_numpy(image_pc)
#
#     image_torch = torch.from_numpy(image_pc).type(torch.FloatTensor)
#     image_unsq = image_torch.unsqueeze_(0)
#     model.to(device)
#     image_model = image_unsq.to(device)
#     logps = model.forward(image_model)
#     ps = torch.exp(logps)
#     top_probes, top_classes = ps.topk(5, dim=1)
#
#     return top_probes, top_classes, image_show
#     # TODO: Implement the code to predict the class from an image file
#
# # # TODO: Display an image along with the top 5 classes
#
# # folder = os.listdir(test_dir)
# # print(os.listdir(test_dir+"/"+folder[random.randint(0, len(folder)-1)]))
#
# # image_path = test_dir
# # ImageFolder = os.listdir(image_path)
# # ImageList = os.listdir(image_path + "/" + ImageFolder[random.randint(0, len(ImageFolder) - 1)])
# # loadedImages = []
# # for image in ImageList:
# #     print(image)
#
# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)
#
# # import json
#
# # with open('cat_to_name.json', 'r') as f:
# #     cat_to_name = json.load(f)
#
# # print(cat_to_name)
#
# # TODO: Display an image along with the top 5 classes
# probs, classes, image = predict(test_dir, model)
#
# indexes = []
# for index in classes[0]:
#     indexes.append(str(int((index))))
#
# nums = []
# for index in probs[0]:
#     nums.append(float((index)))
#
# i = 0
# for index in nums:
#     if index == probs.max():
#         max_index = indexes[i]
#     i += 1
#
# axs = imshow(image, ax=plt)
# axs.axis('off')
# axs.title(cat_to_name[max_index])
# axs.show()
# plt.figure(figsize=(3, 3))
#
# names = []
# for index in indexes:
#     names.append(cat_to_name[index])
#
# y_pos = np.arange(len(names))
# performance = np.array(probs)[0]
# plt.barh(y_pos, performance, align='center',
#          color='blue')
# plt.yticks(y_pos, names)
# plt.gca().invert_yaxis()