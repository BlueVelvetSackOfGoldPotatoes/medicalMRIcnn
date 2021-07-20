import os
import sys

def check_folder_empty(dir_name):
    ''' Terminates program if there s an attempt at
      opening a folder that doesn t exist or at writting
      in a folder that isn't empty.
    '''
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            print("Directory " + dir_name + " is empty, proceding...")
        else:    
            print("Directory " + dir_name + "  is not empty! Terminating...")
            sys.exit()
    else:
        print("Given directory " + dir_name + " doesn't exist: Quitting...")
        sys.exit()