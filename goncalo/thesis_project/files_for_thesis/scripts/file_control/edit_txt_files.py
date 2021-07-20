''' Methods that control file edition
'''

import file_control.folder_empty as folder_checker

def clear_files(files):
    ''' Clears files.
    '''
    for file in files:
        f = open(file,"r+")
        f.truncate(0)
        f.close()

def write_ouput_to_file(output, file):
    ''' Writes output to files.
    '''
    # If folder isn't empty stop the program. 
    folder_checker.check_folder_empty(file)

    with open(file ,"a+") as f:
        f.write(output + "\n")