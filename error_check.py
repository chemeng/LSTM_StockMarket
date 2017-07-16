###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

############################################################
########## This script handles error checking  #############
############################################################

# import std modules
import pybel, glob, os, re, argparse, sys, random, time
from Scripts.globalvars import *
##############################################
### function to print available geometries ###
##############################################
def check_file_exists(file):
    # read from csv
    return os.path.isfile(file)

def check_file_exists_fatal(file):
    if not check_file_exists(file):
        msg = '\nFile ' + file + ' does not exist. Exiting..\n'
        print_msg(msg)
        time.sleep(3)
        sys.exit()
    else:
        return True

def check_dir_exists_fatal(model_dir):
    if not os.path.isdir(model_dir):
        msg = '\nDirectory ' + model_dir + ' does not exist. Exiting..\n'
        print_msg(msg)
        time.sleep(3)
        sys.exit()
    else:
        return True
