''' Methods that control file edition
'''

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
    with open(file ,"a+") as f:
        f.write(output + "\n")