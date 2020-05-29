#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:                                   
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args():
    """
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_directory', type = str, default = 'flowers/', help = 'path to the folder of flowers')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'type of CNN algorithm')
    parser.add_argument('--learning_rate', type = float, default = 0.004, help = 'learning_rate')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'a hidden layer')
    parser.add_argument('--epochs', type = int, default = 10, help = 'one take of learning')
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'using gpu')
 
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    in_args = parser.parse_args()

    return in_args