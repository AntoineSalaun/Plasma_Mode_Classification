#!/usr/bin/env python

# data packages

# our own libraries
from functions_definition import add_extra_features, count_samples, create_samples, split_data,create_tensors, count_each_class
from CNN_model import train_model, compute_nb_errors, Net

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rc_file_defaults() 
##########################################################
#### data importing 
parquet_file = 'TCV_LHD_db4ML.parquet.part'
df = pd.read_parquet(parquet_file, engine ='auto')
##########################################################
#### removing spurious data
mask = df['LDH'] == 'Ip<Ip_MIN'
df_filter = df.drop(index = df[mask].index) #remove Ip<Ip_MIN values 

df_filter = df_filter.dropna() #remove Nan values
df_filter = df_filter.reset_index(drop=True) #reset indexing
df_filter.LDH = df_filter.LDH.cat.remove_categories('Ip<Ip_MIN') #remove Ip<Ip_MIN category

discard_data = len(df.index) - len(df_filter.index) # number of data point that do not contain useful information
print('number of useless data points: ', discard_data)
print('size of filtered data set: ', len(df_filter.index))
print('size of original data set: ', len(df.index))

##########################################################
#### general parameters
window_size = 40 # height of the sample
mini_batch_size = 1000 # mini batch size for the CNN
nb_hidden = 5 # number of neurons in the hidden layers
nb_epochs = 30 # total epochs
ratio = 0.1
torch.manual_seed(0) # for reproductibility

## add extra features to data frame
extra_features = ['fourier_PD']
add_extra_features(df_filter, extra_features)

## features in original data frame after adding the Fourier column
# keys in data frame: time, IP, PD, FIR, WP, LDH, pulse, fourier_PD
features_mask = np.array([False, True, True, False, True, False, False, True ])
features_keep = df_filter.keys().to_numpy()
features_keep = features_keep[features_mask]
width_picture =int( (features_mask+0).sum() ) #total number of features to keep

## compute total samples of the given size (window_size, width_picture)
total_samples, num_pulses = count_samples(df_filter, window_size)
print('total number of samples of the given window size: ', total_samples)
print('total number of pulses: ', num_pulses)

all_samples, all_labels = create_samples(df_filter, total_samples, num_pulses, window_size , width_picture, features_mask)
print('shape of all_samples: ', all_samples.shape)

# create of train and test set
x_train, y_train, x_test, y_test = split_data(all_samples, all_labels, ratio)

# create tensor from numpy arrays
x_train, y_train, x_test, y_test = create_tensors(x_train, y_train, x_test, y_test, mini_batch_size, window_size, width_picture)
samples_train = int(x_train.shape[0])
samples_test = int( x_test.shape[0])
print('samples for train: ', samples_train)
print('samples for test: ', samples_test)

# counting the number of each class that we have in our train set
total_L, total_D, total_H, total_transition = count_each_class(y_train)
print('class 1 total percent: ', total_L*100)
print('class 2 total percent: ', total_D*100)
print('class 3 total percent: ', total_H*100)
print('transition total percent: ', total_transition*100)

#######################################################################
## model training
acc_loss_vector = torch.zeros(nb_epochs) 
model = Net(nb_hidden)
print('training started')
model.train()
train_model(model, x_train, y_train, mini_batch_size, acc_loss_vector, nb_epochs)
print('training finished')

#######################################################################
## plot loss function
nb_train_errors = compute_nb_errors(model, x_train, y_train, mini_batch_size) / x_train.size(0) * 100
print('train error: ', str(nb_train_errors) + '%')

nb_test_errors = compute_nb_errors(model, x_test, y_test, mini_batch_size) / x_test.size(0) * 100
print('test error: ' +str(nb_test_errors) + '%')

plt.plot( torch.arange(1, nb_epochs+1), acc_loss_vector, marker='x', c = 'k');
plt.title('train loss');
plt.xlabel('epoch');
plt.ylabel('loss');
plt.grid(True);
plt.show();