###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

####################################################
#########   Defines class of global    #############
########   variables that are shared   #############
##########    within the program       #############
####################################################

import os, inspect, glob, platform, sys, subprocess,time
from math import sqrt 

########################################
### module for running bash commands ###
########################################
def mybash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        stdout.append(line)        
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

###############################
### global vars placeholder ###
###############################
class globalvars:
    def __init__(self, cmodel=False):
        #############################
        ### PREDICTION PARAMETERS ###
        #############################
        #### prediction window #####
        self.prediction_window = 12 # how many time steps ahead we predict
        #### look back window #####
        self.look_back = 60 # how many time steps in the past to use for prediction
        #### training days ####
        self.training_days =  36*12 # how many of the available months to use for training/testing
        #### maximum training points
        self.max_training_stocks = 500
        #######################################
        ### Define file and directory names ###
        #######################################
        #### current model name ####
        if cmodel:
            self.cmodel = cmodel
        else:
            self.cmodel = '10101020'
        #### current working dir ###
        self.rundir = os.getcwd()
        #### INSTALLATION DIR ####
        self.installdir = '.'
        ###### PROGRAM NAME ######
        self.PROGRAM = 'Algo LSTM'
        #### MODELS filename #####
        self.MODEL = 'Models/'+self.cmodel+'/lstm_model'
        #### SCALERS filename ####
        self.SCALERS = 'Models/'+self.cmodel+'/scalers.pkl'
        #### OUTPUT filename #####
        self.OUTPUT = 'RESULTS'+'/'+self.cmodel
        #### TRAINING FILE NAME ####
        self.TRAINING = 'Stocks_final/'+self.cmodel


##############################
### gui print on msg board ###
##############################
def print_msg(msg):
    print msg
