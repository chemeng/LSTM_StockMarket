###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

################################################################
########## This script handles data visualization  #############
################################################################

# import std modules
import pybel, glob, os, re, argparse, sys, random, joblib
import numpy as np
import datetime as dt
from Scripts.error_check import *
from Scripts.globalvars import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from keras.utils import plot_model

###########################################
############## PLOT NETWORK ###############
###########################################
def plot_network(model):
    plot_model(model, to_file='lstm_network.png', show_shapes=True, show_layer_names=True)
    return True

###########################################
################ PLOT LOSS ################
###########################################
def plot_history(history,globs):
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (MAE)')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.grid(True)
    plt.savefig(globs.OUTPUT+"/MAE_t.png")
    #plt.show()
    plt.clf()
    return True

#########################################
########### PLOT PREDICTION #############
#########################################
def plot_prediction(y_act, y_pred,x_dates,stocks,flag,globs):
    # loop over stocks and plot results
    x = [dt.datetime.strptime(str(d),'%Y-%m-%d').date() for d in x_dates]
    for i in range(len(stocks)):
        y_a = y_act[i,:,0]
        y_p = y_pred[i,:,0]
        stocklabel = stocks[i]
        # print one stock prediction
        plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        plt.plot(x,y_a)
        plt.plot(x,y_p)
        plt.gcf().autofmt_xdate()
        if flag==0:
            plt.title('Test Set Model Prediction')
        elif flag==1:
            plt.title('Train Set Model Prediction')
        plt.ylabel('1-YR Return GVKEY:'+stocklabel)
        plt.xlabel('Year')
        plt.legend(['actual', 'prediction'], loc='upper right')
        plt.grid(True)
        if flag==0:
            plt.savefig(globs.OUTPUT+'/'+stocklabel+"_test.png")
        elif flag==1:
            plt.savefig(globs.OUTPUT+'/'+stocklabel+"_train.png")
        #plt.show()
        plt.clf()
    return True


