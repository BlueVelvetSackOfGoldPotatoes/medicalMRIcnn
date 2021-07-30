import os
import sys

def check_folder_empty(path):
    ''' Terminates program if there s an attempt at
      opening a folder that doesn t exist or at writting
      in a folder that isn't empty.
    '''
    dir_name = os.path.dirname(path)
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        print("Directory " + dir_name + " exists, proceeding...")
        gate = input("File " + file_path + " is not empty! Are you sure you want to procceed? \n y: proceed \n o: exit")
        if gate == 'o':
            print('exiting')
            sys.exit()
        elif gate == 'y':
            print('Continuing...')
            return
        while gate != 'y':
            gate = input('...\n y: proceed \n o: exit')    
    else:
        print("Given directory " + dir_name + " doesn't exist: Quitting...")
        sys.exit()