#!/usr/bin/env python
###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

#####################################################
############  Script that coordinates  ##############
##############  start of LSTM algo   ################
#####################################################

import sys, argparse, os, platform
from Scripts.dataprep import *
from Scripts.train import *
from Scripts.predict import *
from Scripts.error_check import *
from Scripts.globalvars import *

def run_lstm(args,globs):
    ############################
    ######## INPUT ARGS ########
    ############################
    if '-h' in args:
        parser.print_help()
        exit()
    ###########################
    ###### LSTM TRAINING ######
    ###########################
    if args.t:
        ### get y, X, prediction labels and factor labels in numpy format
        X3d, y3d, T, stocks, X_labels, dates = data_preprocess_train(globs)
        ### build LSTM model
        acc = build_lstm_rnn(X3d, y3d, stocks, X_labels, dates, globs)
        msg = "\n### Training is over. See your results in the corresponding folder. ###\n"
        print_msg(msg)
    #############################
    ###### LSTM PREDICTION ######
    #############################
    if args.p:
        ### get y, X, prediction labels and factor labels in numpy format
        X3d, y3d, T, stocks, X_labels = data_preprocess_train(True)
        acc = predict_lstm(X3d, y3d, T, stocks, X_labels)
        msg = "\n### Prediction is over. See your results in the corresponding folder. ###\n"
        print_msg(msg)
    ###########################
    ###### LSTM EVALUATE ######
    ###########################
    if args.e:
        ### get y, X, prediction labels and factor labels in numpy format
        y, X, y_lab, X_lab, df = data_preprocess_train(eval=True)
        ### predict
        y_predicted = predict_lstm_test(X, y_lab)
        ### get right labels
        y_actual = df[df.columns[0]]
        ### write labels
        write_labels(y_predicted, df, globs)
        overall_acc = lstm_accuracy(y_predicted, y_actual)
        msg = "\n### Overall accuracy was: "+'{:.2f}'.format(100*overall_acc)+"% ###\n"
        print_msg(msg)
        
        
