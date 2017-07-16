###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis   
###########################################################

# import std modules
import pandas as pd
import numpy as np
import joblib, sys, keras
from Scripts.error_check import *
from Scripts.globalvars import *
from Scripts.dataprep import *
from Scripts.train import *
from Scripts.visualize import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide messy TensorFlow warnings

##############################################
### function to coordinate LSTM prediction ###
##############################################
def predict_lstm(X3d, y3d, T, stocks, X_labels):
    msg = "\nPredicting variables.."
    print_msg(msg)
    ### get globalvars
    globs = globalvars()
    ### load model
    json_file = open(globs.MODEL+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(globs.MODEL+'.h5')
    # load data scalers
    scalerX = joblib.load(globs.SCALERS)
    # load how many steps are used in prediction
    timesteps = globs.look_back
    # get relevant info
    num_stocks = len(X3d[:,0,0])
    num_features = len(X3d[0,0,:])
    # shape correctly for prediction
    X, y = unfold_samples(X3d,y3d)
    # scale correctly
    X_scaled, scalerX = scale_data(X,scalerX)
    # print message
    msg = "Predicting for model "+globs.cmodel+"\n"
    print_msg(msg)
    # predict
    y_pred = model.predict(X_scaled)
    # actual number of predicted timesteps
    num_prediction_timesteps = len(X3d[0,:,0])-timesteps 
    if num_prediction_timesteps < 0:
        print "You need at least "+timesteps+" timesteps to predict the next one."
        exit()
    # shape correctly for post-proc
    y_act, y_pred = extract_predicted(y, y_pred, num_stocks)
    # calculate how many times we had right sign
    accuracies = get_accuracy(y_pred,y_act)
    # plot the stock results
    p = plot_prediction(y_act, y_pred, stocks,0)
    # write results
    w = write_results(y_pred,y_act,X_labels[0],stocks,timesteps,accuracies)
    msg = "\nEnd of prediction. Results written in corresponding files\n"
    print_msg(msg)
    return accuracies

###############################
#### GET ACCURACY OF MODEL ####
###############################
def get_accuracy(y_pred,y_act):
    # get number of stocks
    num_stocks = len(y_pred[:,0,0])
    # get prediction timesteps
    num_timesteps = len(y_pred[0,:,0])
    # initialize accuracy matrix
    accuracies = np.zeros((num_stocks,1))
    ### calculate accuracy as
    ### return_t = 1 + sign(R_t^actual*R_t^prediction)*|R_t^actual|
    for i in range(num_stocks):
        cum_log_return = 0 
        for t in range(num_timesteps):
            R_actual = np.exp(y_act[i,t,0])-1
            R_pred = np.exp(y_pred[i,t,0])-1
            log_return_t = 1+np.sign(R_actual*R_pred)*np.abs(R_actual)
            cum_log_return += np.log(log_return_t)
        accuracies[i] = cum_log_return
    return accuracies

###########################
#### GET RMSE OF MODEL ####
###########################
def get_rmse(y_pred,y_act):
    # get number of stocks
    num_stocks = len(y_pred[:,0,0])
    # get prediction timesteps
    num_timesteps = len(y_pred[0,:,0])
    # initialize accuracy matrix
    rmse = np.zeros((num_stocks,1))
    ### calculate RMSE error
    for i in range(num_stocks):
        rmse[i] = np.linalg.norm(y_act[i,:,0]-y_pred[i,:,0])/len(y_act[i,:,0])
    return np.mean(rmse)


################################
#### EXTRACT PREDICTED DATA ####
################################
def extract_predicted(y_actual, y_pred, num_stocks):
    num_of_samples = len(y_pred[:,0,0])
    num_predictions_per_stock = (num_of_samples/num_stocks)-1
    # initialize placeholders
    y_final_act = np.zeros((num_stocks,num_predictions_per_stock,1))
    y_final_pred = np.zeros((num_stocks,num_predictions_per_stock,1))
    n = 0 ;
    for i in range(0,num_stocks):
        for t in range(0,num_predictions_per_stock):
            y_final_act[i,t,0] = y_actual[n,-1,0]
            y_final_pred[i,t,0] = y_pred[n,-1,0]
            n += 1
    # return predicted of shape [num_stocks,timesteps]
    return y_final_act, y_final_pred

###########################
###### WRITE RESULTS ######
###########################
def write_results(y_pred,y_act,label,stocks,dates,accuracies,globs,train=False):
    # loop over stocks and write one file at a time
    for i in range(0,len(y_pred[:,0,0])):
        ### combine data to pandas dataframe    
        # predicted values
        y_p_df = pd.DataFrame(data=y_pred[i,:,0],columns=['Prediction'])
        # actual values
        y_a_df = pd.DataFrame(data=y_act[i,:,0],columns=[label])
        # combine dataframes
        yap_df = pd.concat([y_p_df, y_a_df],axis=1,join='outer')
        # change index to correct timestep (timesteps+1)
        yap_df.index = dates
        ### write into csv
        if train:
            yap_df.to_csv(globs.OUTPUT+'/'+stocks[i]+'_train_predicted.csv',index=True,index_label="Date") 
        else:
            yap_df.to_csv(globs.OUTPUT+'/'+stocks[i]+'_test_predicted.csv',index=True,index_label="Date") 
        ### write down accuracies
    acc_df = pd.DataFrame(data=accuracies,columns=['Cumulative Return'])
    acc_df.index = stocks
    acc_df.to_csv(globs.OUTPUT+'/'+'accuracy_predicted.csv',index=True,index_label="StockID") 
    return 1

