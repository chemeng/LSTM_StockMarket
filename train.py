###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

#########################################################
########## This script handles SVM training  ############
#########################################################

# import std modules
import pandas as pd
import numpy as np
import joblib, sys, keras
from Scripts.error_check import *
from Scripts.dataprep import *
from Scripts.globalvars import *
from Scripts.visualize import *
from Scripts.predict import * 
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide messy TensorFlow warnings

############################################
### function to coordinate LSTM training ###
############################################
def build_lstm_rnn(X3d, y3d, stocks, X_labels, dates, globs):
    ##########################
    ###### ARCHITECTURE ######
    ##########################
    layer_lstm0_neurons = 64
    layer_lstm1_neurons = 64
    layer_lstm2_neurons = 32
    layer_den1_neurons = 64
    layer_denseF_neurons = 1
    #########################
    ##### PREPARE DATA ######
    #########################
    splitfactor = 0.8 # how much of the data will be training/validation
    ### get model size ###
    # number of stocks
    num_stocks = len(X3d[:,0,0])
    # number of features
    num_features = len(X3d[0,0,:])
    # number of timesteps
    num_timesteps = len(X3d[0,:,0])
    # split to training/test data
    X_train, y_train, X_test, y_test = split_data(X3d, y3d, splitfactor)
    ### standardize data
    # scaler acts on arrays of size (num_timesteps,num_features)
    # thus for each stock separately
    X_train, scalerX = scale_data(X_train)
    # save scalers on file
    joblib.dump(scalerX, globs.SCALERS)
    # convert to training samples based on look back
    X_train, y_train = unfold_samples(X_train,y_train)
    #### X has to be 3D: (num_stocks, num_look_back, num_features) ####
    #### Y has to be 3D: (num_stocks, num_look_back, num_targets)  ####
    #########################
    ####### TRAINING ########
    #########################
    training_batch_size = 64 # how many stocks to include in each batch
    training_epochs = 200
    window_length = len(X_train[0,:,0])
    backprop_algo = 'rmsprop'
    early_stop_epochs = 10
    ###########################
    ##### Build the model #####
    ###########################
    ### expected input data shape: (num_stocks, timesteps, data_dim) ###
    # initialize models
    if glob.glob(globs.MODEL+'.json') and glob.glob(globs.MODEL+'.h5'):
        # load json and create model
        json_file = open(globs.MODEL+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(globs.MODEL+'.h5')
    else:
        model = Sequential()
        # add 1st LSTM layer
        model.add(LSTM(layer_lstm0_neurons,return_sequences=True,
               input_shape=(len(X_train[0,:,0]), num_features),kernel_initializer='random_normal',
               recurrent_initializer='random_normal',bias_initializer='zeros',
               dropout=0.4, recurrent_dropout=0.2,name='lstm0'))  # returns a sequence of vectors of dimension layer_lstm0_neurons
        # add 2nd LSTM layer
        #model.add(LSTM(layer_lstm1_neurons,return_sequences=True,
        #    kernel_initializer='random_normal',recurrent_initializer='random_normal',bias_initializer='zeros',
        #    dropout=0.4, recurrent_dropout=0.3,name='lstm1')) # returns a sequence of vectors of dimension layer_lstm1_neurons
        # add 1st Dense layer with PReLU activation
        # TimeDistributed makes the cost function be calculated at each time step, not only at the end
        model.add(TimeDistributed(Dense(layer_den1_neurons,
            kernel_initializer='random_normal',bias_initializer='zeros',name='dense0'))) 
        # batch normalization for faster algo
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # add dropout layer for regularization
        model.add(Dropout(0.4))
        # add Final Dense layer
        model.add(TimeDistributed(Dense(layer_denseF_neurons,
            kernel_initializer='random_normal',bias_initializer='zeros',name='denseF')))
        model.add(Activation('linear'))
    # build model
    model.compile(loss='mean_squared_error',optimizer=backprop_algo,metrics=['mae'])
    print "Model summary"
    print model.summary()
    ###########################
    ### Print runtime info ####
    ###########################
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    ###########################
    ##### Train the model #####
    ###########################
    # early stopping
    earlyStopping=EarlyStopping(monitor='val_loss', patience=early_stop_epochs, verbose=1, mode='auto')
    # epoch counter
    n_epochs = 0 
    # print info
    msg = "Training for model "+globs.cmodel+" for max epochs: "+str(training_epochs)+"\n"
    print_msg(msg)
    # training
    history = model.fit(X_train, y_train, batch_size=training_batch_size,epochs=training_epochs,
                callbacks=[earlyStopping],verbose = 1,validation_split=0.2,shuffle=True)
    # save model
    p = save_lstm(model,globs)
    # get actual epochs trained
    num_epochs_trained = training_epochs # to be fixed
    # augment epoch counter
    n_epochs += num_epochs_trained
    ###########################
    ###### Evaluate model #####
    ###########################
    # shape correctly for prediction
    X_test,y_test = unfold_samples(X_test,y_test)
    # scale correctly
    X_test, scalerX = scale_data(X_test,scalerX)
    # predict
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # shape correctly for post-proc
    y_act, y_pred = extract_predicted(y_test, y_pred,num_stocks)
    y_act_train, y_pred_train = extract_predicted(y_train, y_pred_train,num_stocks)
    # fix dates
    training_timesteps = y_pred_train.shape[1] # grab how many timesteps trained
    test_timesteps = y_pred.shape[1] # grab how many timesteps tested
    total_timesteps = len(dates)
    dates_train = dates[globs.look_back:globs.look_back+training_timesteps]
    dates_test = dates[total_timesteps-test_timesteps:]
    # calculate cumulative return for accuracy
    accuracies = get_accuracy(y_pred,y_act)
    # calculate RMSE
    rmse_test = get_rmse(y_pred,y_act)
    rmse_train = get_rmse(y_pred_train, y_act_train)
    msg = "\nRMSE for training set is "+"{:.3E}".format(rmse_train)+" and for test set is "+"{:.3E}".format(rmse_test)
    print_msg(msg)
    # write to file
    with open("RMSE.txt", "a") as myfile:
        msg = globs.cmodel+" Stocks:"+str(num_stocks)+" Train:"+"{:.3E}".format(rmse_train)
        msg += " Test:"+"{:.3E}".format(rmse_test)+"\n"
        myfile.write(msg)
    # plot training and validation accuracy (MAE)
    if history.history.keys():
        p = plot_history(history,globs)
    # write results
    w = write_results(y_pred_train,y_act_train,X_labels[0],stocks,dates_train,accuracies,globs,True)
    w = write_results(y_pred,y_act,X_labels[0],stocks,dates_test,accuracies,globs)
    # plot accuracies stock
    p = plot_prediction(y_act_train, y_pred_train, dates_train, stocks,1,globs)
    p = plot_prediction(y_act, y_pred, dates_test, stocks,0,globs)
    # plot network
    p = plot_network(model)
    return accuracies

    ###########################
    ####### SAVE MODEL ########
    ###########################
def save_lstm(model,globs):
    # serialize model to JSON
    model_json = model.to_json()
    with open(globs.MODEL+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(globs.MODEL+'.h5')
    return True
