###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

#############################################################
########## This script handles data processing  #############
#############################################################

# import std modules
import pybel, glob, os, re, argparse, sys, random, joblib
import pandas as pd
import numpy as np
from numpy import newaxis
from Scripts.error_check import *
from Scripts.globalvars import *
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from fancyimpute import KNN

#############################################
### function to coordinate pre-processing ###
#############################################
def data_preprocess_train(globs,predict=False):
    msg = "Preparing data.."
    print_msg(msg)
    ### load data
    df_original = data_load(globs)
    ### find correct data types & convert to
    X3d, y3d, T, stocks, X_labels, dates = data_clean_train(df_original)
    ### fill missing data with k-NN imputation
    X3d = data_fill_kNN(X3d, stocks)
    #X3d = np.nan_to_num(X3d)
    ### remove prediction data if training
    if not predict:
        X3d = X3d[:,:globs.training_days,:]
        y3d = y3d[:,:globs.training_days,:]
        # if more stocks than it can handle pick subset randomly
        if (len(stocks) > globs.max_training_stocks):
            X3d = X3d[np.random.choice(X3d.shape[0], globs.max_training_stocks, replace=False), :, :]
            y3d = y3d[np.random.choice(y3d.shape[0], globs.max_training_stocks, replace=False), :, :]
            msg = "Data is too big. Random subset of " + str(globs.max_training_stocks)+" selected."
            print_msg(msg)
    else:
        ### remove trained data if predicting
        X3d = X3d[:,(globs.training_days-globs.look_back):,:]
        y3d = y3d[:,(globs.training_days-globs.look_back):,:]
    ### print statistics
    msg = "Data includes "+str(len(stocks))+" stocks for a total of "+str(len(y3d[0,:]))+" time points"
    msg += " and "+str(len(X3d[0,0,:]))+" features:\n\n"+",".join(X_labels)+"."
    msg += "\n\nUsing a look back window of "+str(globs.look_back)+" steps"
    msg += " and a forward prediction window of "+str(globs.prediction_window)+" steps.\n"
    print_msg(msg)
    msg = "Data pre-processing complete.\n"
    print_msg(msg)
    return X3d, y3d, T, stocks, X_labels, dates


#############################################
### function to read data frame from file ###
#############################################
def data_load(globs):
    datafolder = globs.TRAINING
    # initialize list placeholder dict for dataframes
    df = dict()
    check_dir_exists_fatal(datafolder) # check file exists
    ### loop over all csv files to build list of dataframes
    for file in sorted(os.listdir(datafolder)):
        if file.endswith(".csv"):
            # read csv file
            file_id = file[:-4] # grab stock_id
            df[file_id] = pd.read_csv(os.path.join(datafolder, file))

    return df

##############################
### function to clean data ###
##############################
def data_clean_train(df):
    globs = globalvars()
    # extract model size
    num_stocks = len(df) # different stocks
    num_timesteps = df.values()[0].shape[0] # total timesteps available
    num_features = df.values()[0].shape[1] # total features
    # feature labels (not included)
    X_labels = df.values()[0].columns[1:]
    dates = df[df.keys()[0]].iloc[:,0] # grab dates
    # remove timestamp and ID
    for key in df.keys():
        df[key].drop(df[key].columns[[0]], axis=1, inplace=True)
        # make time axis start at 1 instead of 0
        df[key].index += 1
    # convert into 3D panel
    df3d = pd.Panel(df)
    '''
    ##############
    # old way to convert stacked 2D dataframe to 3D panel
    ### convert stacked/record format Date-Stock-Features into a 3D Panel
    df3d = pd.Panel(dict(zip(X_labels, [df.pivot(index=df.columns[0], columns=df.columns[1], values=i) for i in X_labels])))
    # reset panel labeling (alphabetic by default)
    df3d = df3d.reindex_axis(X_labels, axis=0)
    ##############
    '''
    # extract stocks
    stocks = list(df3d.axes[0])
    # extract timesteps
    T = list(df3d.axes[1])
    # extract features
    features = list(df3d.axes[2])
    ### convert to corresponding numpy 3D ndarray ###
    #  keras input: X3d[stock, timestep, feature]
    X3d = df3d.values
    # extract target -> returns 3D: y3d=[stock,timestep] shifted by prediction_window
    y3d = X3d[:,globs.prediction_window:,0]
    y3d = y3d[:,:,newaxis] # forces 3D
    # remove last observation from X3d (last obs, 2nd axis)
    X3d = X3d[:,:(num_timesteps-globs.prediction_window),:]
    dates = dates[:(num_timesteps-globs.prediction_window)]
    return X3d, y3d, T, stocks, X_labels, dates

##########################################
### function to perform kNN imputation ###
##########################################
def data_fill_kNN(X,stocks):
    ### fill in using kNN with 3rd nearest neighbor
    # loop over dataframes (one per stock)
    for i in range(0,len(stocks)):
        try:
            X[i,:,:] = KNN(k=3).complete(X[i,:,:])
            print "Input dataframe for stock "+stocks[i]+" imputed with k-NN.\n"
        except ValueError:
            print "Input dataframe for stock "+stocks[i]+" has no missing entries.\n"
    ### fill most frequent value # OLD
    #X = X.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return X

##############################
##### STANDARDIZE INPUT ######
##############################
def scale_data(X,scalerX=None):
    # reshape X for accounting for all samples
    num_stocks = len(X[:,0,0])
    num_timesteps = len(X[0,:,0])
    num_features = len(X[0,0,:])
    if scalerX is None:
        # flatten array to aggregate different observations
        X_reshaped = X.reshape(num_stocks*num_timesteps,num_features)
        scalerX = StandardScaler().fit(X_reshaped)
    # apply scaler for each stock
    X_scaled = np.zeros((num_stocks,num_timesteps,num_features))
    for i in range(0,num_stocks):
        X_scaled[i,:,:] = scalerX.transform(X[i,:,:])
    return X_scaled,scalerX

###############################
##### INVERSE STANDARDIZE #####
###############################
def scale_back_data(X,scalerX=None):
    # reshape X for accounting for all samples
    num_stocks = len(X[:,0,0])
    num_timesteps = len(X[0,:,0])
    num_features = len(X[0,0,:])
    # apply scaler for each stock
    X_scaled = np.zeros((num_stocks,num_timesteps,num_features))
    for i in range(0,num_stocks):
        X_scaled[i,:,:] = scalerX.inverse_transform(X[i,:,:])
    return X_scaled,scalerX



#####################################
#### CONVERT SEQUENCE TO SAMPLES ####
#####################################
def unfold_samples(X_in,y_in):
    globs = globalvars()
    # get number of stocks
    num_stocks = len(X_in[:,0,0])
    # get total time steps available in training set
    num_timesteps = len(X_in[0,:,0])
    # get features
    num_features = len(X_in[0,0,:])
    # get look back timesteps
    num_look_back = globs.look_back
    # get total samples for all stocks and all times
    num_of_samples = num_stocks*(num_timesteps-num_look_back+1)
    # create numpy 3D array of samples per stock
    X = np.zeros((num_of_samples,num_look_back,num_features))
    # create numpy 2D array of predictions
    y = np.zeros((num_of_samples,num_look_back,1))
    # loop to create samples
    n = 0
    for i in range(num_stocks):
        for t in range(num_timesteps-num_look_back+1):
            X[n,:,:] = X_in[i,t:(t+num_look_back),:]
            y[n,:,0] = y_in[i,t:(t+num_look_back),0]
            n += 1
    return X,y


###########################################
############### SPLIT DATA ################
###########################################
def split_data(X,y,splitfactor=0.8):
    globs = globalvars()
    # get look back
    num_look_back = globs.look_back
    # split training and test data
    no_obs_training = int(splitfactor*len(X[0,:,0]))
    X_train = X[:,:no_obs_training,:]
    y_train = y[:,:no_obs_training,:]
    X_test = X[:,(no_obs_training-num_look_back):,:]
    y_test = y[:,(no_obs_training-num_look_back):,:]
    return X_train, y_train, X_test, y_test

