''' Methods that control file edition
'''
import sys
import os
from .folder_empty import *

def check_file_empty(file_path):
    ''' Terminates program if there s an attempt at
      overwriting a file that doesn t exist or if file is not empty.
    '''
    if os.path.exists(file_path):
        print("File " + dir_name + " exists, proceding...")
        if os.stat(file_path).st_size != 0:
            gate = input("File " + file_path + "  is not empty! Are you sure you want to procceed? \n y: proceed \n o: exit")
            # Exit
            if gate == 'o':
                print('exiting')
                sys.exit()
            # Is ok
            elif gate == 'y':
                print('Continuing...')
                return
            while gate != 'y':
                gate = input('...\n y: proceed \n o: exit')
    else:
        # Exit
        print("Given file " + file_path + " doesn't exist: Quitting...")
        sys.exit()

def clear_files(files):
    ''' Clears files.
    '''
    for file in files:
        f = open(file,"r+")
        f.truncate(0)
        f.close()

def write_output(output, file):
    ''' Writes output to files.
    '''
    # Check if folder empty and exists
    folder_empty.check_folder_empty(file)
    # Check if file empty and exists
    check_file_empty(file)

    with open(file ,"a+") as f:
        f.write(output + "\n")