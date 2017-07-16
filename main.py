#!/usr/bin/env python
###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis & Arman Rezaee  
###########################################################

##########################################################
############  Main script that coordinates  ##############
#############  all parts of the program   ################
##########################################################

import sys, argparse, os, platform
import json, urllib2
from Scripts.dataprep import *
from Scripts.train import *
from Scripts.predict import *
from Scripts.error_check import *
from Scripts.run_lstm import *
from Scripts.globalvars import *

if __name__ == '__main__':
    globs = globalvars()
    ### run GUI by default ###
    args_init = sys.argv[1:]
    ### create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--d",help="model directory")
    parser.add_argument("-t","--t",help="train model, needs train_data.csv",action='store_true')
    parser.add_argument("-p","--p",help="predict labels, needs predict_data.csv",action='store_true')
    parser.add_argument("-e","--e",help="evaluate accuracy, needs eval_data.csv",action='store_true')
    args=parser.parse_args()
    ###############################
    ###### PRINT WELCOME MSG #######
    ################################
    msg =  "\n**************************************************************"
    msg += "\n********** Welcome to "+globs.PROGRAM+". Let's get started! **********\n"
    msg += "**************************************************************\n"
    print msg
    # create folders
    if args.d:
        globs = globalvars(args.d)
    if not os.path.exists(globs.MODEL):
        os.makedirs(globs.MODEL)
    if not os.path.exists(globs.OUTPUT):
        os.makedirs(globs.OUTPUT)
    ##################################
    ######## RUN LSTM VERSION ########
    ##################################
    run_lstm(args,globs)
    ###### PRINT GOODBYE MSG #######
    msg =  "\n**************************************************************"
    msg += "\n****** Thank you for using "+globs.PROGRAM+". Have a nice day! *******\n"
    msg += "**************************************************************\n"
    msg += "\nDeveloped by T.Ioannidis\nMassachusetts Institute of Technology, 2017\n"
    print msg
