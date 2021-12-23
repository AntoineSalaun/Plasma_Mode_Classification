#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:44:35 2021

@author: bruno

@description:
    this file contains all auxiliary functions to handle the data set and transoform it
    in such a way pytorch can work with the data
"""

# import of useful libraries
import numpy as np
from scipy import signal
import torch

##############################################################
##############################################################
'''
    Functions for data pre-processing
'''
def balance_subset(samples,labels):    
    """
    @description:
        This function divides the data train set in such a way that we have 
        a balanced data set to train; that is, it returns three equally-sized 
        data sets for each state L, D or H.         
    @parameters:        
        samples (numpy array): train/test data points
        labels (numpy array): contains the samples/data labels
    """
    number_of_L = np.sum(labels, axis=0)
    scale = np.floor( min(number_of_L) )
    L_rows = np.where(labels[:,0]==1)
    D_rows = np.where(labels[:,1]==1)
    H_rows = np.where(labels[:,2]==1)
    
    minimum = min(len(L_rows[0]),len(D_rows[0]),len(H_rows[0]))
    
    sub_L = np.random.choice(L_rows[0] , minimum)
    sub_D = np.random.choice(D_rows[0] , minimum)
    sub_H = np.random.choice(H_rows[0] , minimum)
    
    rows=np.concatenate((sub_L, sub_D,sub_H), axis=None)    
    
    balanced_labels = labels[rows]
    balanced_samples = samples[rows]
    
    return balanced_samples, balanced_labels

##########################################################
## add extra features 
def add_extra_features(data_frame, extra_features):    
    """
    @description:
        This function adds the total number of extra features
        to the clean data frame. It initializes as many columns as 
        required to zero. This is done inplace in the data frame
    @parameters:        
        data_frame (pandas data frame): clean data frame
        extra_features (python list): contains the name of the columns
        to be added to the original data frame
    """
    for feature in extra_features:
        data_frame[feature] = np.zeros((data_frame.pulse.shape))

## compute total samples of the given size (window_size, width_picture)
def count_samples(data_frame, window_size):
    """
    Description:
        This function counts how many complete samples of size (window_size) 
        and the total number of experiments we can obtain from each 
        pulse/experiment 
    Parameters:        
        data_frame (pandas data frame): clean data frame
        window_size (int): size of the continuos samples of the original
        data frame that we use to construct a single sample of the desired 
        size  
    """
    num_pulses=int(max(data_frame['pulse'].values))        
    total_samples = 0
    for k in range(num_pulses):                
        df_experiment = data_frame[data_frame.pulse == k + 1]        
        num_samples = int( df_experiment.shape[0]/window_size )        
        total_samples  += num_samples                
    return total_samples, num_pulses

def create_samples(data_frame, total_samples, num_pulses, window_size , width_picture, features_keep, expand_H=False):
    """
    Description:
        This creates the samples given of the given size from the clean data frame
    Parameters:        
        data_frame (pandas data frame): contains the clean data set
        total_samples (int): how many samples of the desired shape we have
        num_pulses (int): total number of experiments/pulses in the original 
        data set
        window_size (int): as described previously
        width_picture (int): total number of features kept in the samples
        for training 
        features_keep: mask to indicate which features to keep in the samples
        for training 
        expand_H=False: if we want to have more samples for the H states to balance 
        the total number of samples belonging to each class
        of the whole data sample to be kept for testing
        shuffle (boolean): to randomly shuffle the indices of the data
        seed (int): to set the seed of the random generator for reproductibility        
    """
    # to store the samples of the given size, intialize to 0
    all_samples  = np.zeros(total_samples * window_size * width_picture).reshape((-1, window_size, width_picture))
    all_labels = np.zeros(total_samples * 3).reshape((-1, 3))
    counter = 0 
    
    for k in range(num_pulses):    
        #print('creating data set from pulse: ', k+1 )
        mask_experiment = data_frame.pulse == k + 1
        df_experiment = data_frame[mask_experiment]
        df_experiment = df_experiment.reset_index(drop=True)        
                
        # fourier computation on PD feature
        df_feature= df_experiment.PD.copy().to_numpy()
        fs = 1e4
        _, _, Sxx = signal.spectrogram(df_feature, fs,nperseg = 512, noverlap = 64 )                          
        # this will throw a warning since we are writing in the original data frame
        # which is want we want to do
        data_frame.fourier_PD[mask_experiment] = np.resize(np.mean(Sxx, axis=0), df_feature.shape)  
                        
        # labels to numerical values
        maskl = df_experiment.LDH == 'L'
        maskd = df_experiment.LDH == 'D'
        maskh = df_experiment.LDH == 'H'
        labels = np.vstack((maskl, maskd, maskh)).T + 0.0
        
        features_exp = df_experiment.keys().to_numpy()
        mask_features_exp = features_keep
        features_exp = features_exp[mask_features_exp]
        x_exp = np.array( df_experiment.loc[:, features_exp].values )        
        
        # this number varies from one experiment to another  
        num_samples = int( x_exp.shape[0]/window_size )
        step = 0            
        for i in range(num_samples):
            all_samples[i + counter] = x_exp[ step : step + window_size, : ].reshape((-1, window_size, width_picture)) 
            all_labels[i + counter] = labels[ step : step + window_size, : ].mean(axis = 0).reshape((-1, 1, 3))                
            step += window_size            
        counter +=num_samples
        
        # expand H states
        if expand_H: 
            h_state = all_labels == np.array([0.0, 0.0, 1.0]).reshape((-1, 3))
            h_state = (h_state + 0.0).prod(axis=1) > 0
            h_samples = all_samples[h_state]
            id_rows = torch.randperm(h_samples.shape[1]) # random shuffle rows of H samples
            id_columns = torch.randperm(h_samples.shape[2]) # random shuffle columns of H samples
    
            h_samples_shuffled = h_samples[:, id_rows, :]
            h_samples_shuffled = h_samples_shuffled[:, : ,id_columns ]
            y_h_samples = torch.tensor([0.0, 0.0, 1.0]).view(-1, 3).repeat(h_samples_shuffled.shape[0], 1)
                
            all_samples = np.concatenate((all_samples, h_samples_shuffled), axis=0) 
            all_labels = np.concatenate((all_labels, y_h_samples), axis=0)     
        
    return all_samples, all_labels

### split data set into train and test
def split_data(data, labels, ratio, shuffle = True, seed = 0):
    """
    Description:
        This function counts how many complete samples of size (window_size) 
        and the total number of experiments we can obtain from each 
        pulse/experiment 
    Parameters:        
        data (numpy array): contains the samples given in the desired shape
        labels (numpy array): contains the labels of such samples
        ratio (float): number between 0 and 1. It represents the proportion
        of the whole data sample to be kept for testing
        shuffle (boolean): to randomly shuffle the indices of the data
        seed (int): to set the seed of the random generator for reproductibility        
    """
    data_size= data.shape[0]
    indices = list(range(data_size))
    split = int(np.floor(ratio * data_size))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    #train data  set 
    x_train = data[train_indices]
    y_train = labels[train_indices]
    
    #test data set 
    x_test = data[test_indices]
    y_test = labels[test_indices]
    
    return x_train, y_train, x_test, y_test

def create_tensors(data_train, label_train, data_test, label_test, mini_batch_size, window_size, width_picture):
    """
    Description:
        This function transforms the input of the function create_samples to
        pytorch tensors
    Parameters:        
        these are described previously
    """
    samples_train = int(data_train.shape[0]/ mini_batch_size)
    samples_test = int(data_test.shape[0]/ mini_batch_size)
    
    samples_train = int(samples_train * mini_batch_size)
    samples_test = int(samples_test * mini_batch_size)
        
    # transform into a tensor
    x_train = torch.from_numpy(data_train).float().reshape(-1, 1, window_size, width_picture)[: samples_train ]
    y_train = torch.from_numpy(label_train).float()[: samples_train ]
    x_test = torch.from_numpy(data_test).float().reshape(-1, 1, window_size, width_picture)[: samples_test ]
    y_test = torch.from_numpy(label_test).float()[: samples_test ]  
    
    return x_train, y_train, x_test, y_test

def count_each_class(labels):    
    """
    Description:
        This function counts how many samples from each class we have in our
        train or test sets. For informative purpose only
    Parameters:        
        labels (numpy array): labels of the train or test sets        
    """
    label_L = labels == torch.tensor([1., 0., 0.]).view(-1,3)
    label_D = labels == torch.tensor([0., 1., 0.]).view(-1,3) 
    label_H = labels == torch.tensor([0., 0., 1.]).view(-1,3) 
    label_transition = labels < torch.tensor([1., 1., 1.]).view(-1,3)

    label_L = label_L + 0.0
    label_D = label_D + 0.0
    label_H = label_H + 0.0
    label_transition = label_transition + 0.0

    total_L = label_L.prod(dim = 1).mean()
    total_D = label_D.prod(dim = 1).mean()
    total_H = label_H.prod(dim = 1).mean()
    total_transition= label_transition.prod(dim = 1).mean()
    
    return total_L, total_D, total_H, total_transition

##############################################################

