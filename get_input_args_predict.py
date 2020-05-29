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
    parser.add_argument('test_image_path', type = str, default = '/flowers/test', help = 'test_image_path')
    parser.add_argument('checkpoint', type = str, default = 'checkpoint_flower.pth', help = 'model_save')
    parser.add_argument('--top_k', type = int, default = 3, help = 'top probability')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'file of cat_to_name')
    parser.add_argument('--gpu_usage', type = str, default = 'gpu', help = 'using gpu')

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    in_args = parser.parse_args()

    return in_args