#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:43:44 2021
    
@author: bruno
    @description:
    This file contains the CNN model for the project. 
    Each function is described when indicated below.
    Note:
    the compute_nb_errors functions is based partially on the examples found on
    https://fleuret.org/dlc/    
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

######################################################################
def train_model(model, train_input, train_target, mini_batch_size, acc_loss_vector, nb_epochs = 20):    
    """
    @description:
        This function trains the convolutional neuronal network(CNN) for the specified 
        parameters as follows
    @parameters:        
        model: pytorch CNN
        train_input (tensor): contains the samples/data points of the train set
        train_target (tensor): contains the labels of the train set
        mini_batch_size (int): size of mini-batches
        acc_loss_vector(numpy array): to store the train loss value for each epoch        
        nb_epochs(int) : number of total epochs                 
    """
    criterion = nn.CrossEntropyLoss()        
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr = 1e-3,  momentum=0.9)    
    #optimizer = optim.AdamW(model.parameters(),lr=1e-3, betas=(0.9, 0.99), weight_decay=0.05)
        
    for e in range(nb_epochs):
        acc_loss = 0          
        for b in range(0, train_input.size(0), mini_batch_size):            
            output = model(train_input.narrow(0, b, mini_batch_size))            
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).squeeze())
            acc_loss = acc_loss + loss.item()            
            model.zero_grad()
            loss.backward()
            optimizer.step()
        acc_loss_vector[e] = acc_loss
        print('epoch: ' +  str(e) + ', loss: ' + str(acc_loss) )

def compute_nb_errors(model, data_input, data_target, mini_batch_size):    
    """
    @description:
        This function is used to compute the error for train and test sets. 
        We compare the index of the max value of the true label to the index
        of the output of the network. If these are equal, the data sample is 
        correctly classified
    @parameters:        
        model: pytorch CNN that is already train
        data_input (tensor): contains the samples/data points of the train/test set
        data_target (tensor): contains the labels of the train/test set
        mini_batch_size (int): size of mini-batches
        acc_loss_vector(numpy array): to store the train loss value for each epoch
              
    """    
    nb_data_errors = 0    
    with torch.no_grad():
        for b in range(0, data_input.size(0), mini_batch_size):
            output = model(data_input.narrow(0, b, mini_batch_size))    
            _, predicted = torch.max(output, 1)    
            for k in range(mini_batch_size):
                if data_target[b + k].view(-1, 3).max(1)[1]  != predicted[k] :
                    nb_data_errors = nb_data_errors + 1
    return nb_data_errors

##########################################################
class Net(nn.Module):    
    '''
    Description:
        This model consist of one convolutional layer to extract features. This layer
        is composed of three convolutional layers, each followed by max pooling, 
        batch normalization, and a relu function
        The second part of the model corresponds to the dense layers. In our case,
        this consists of 5 hidden layers with the specified number of nueorons each.
        At the output of the network, we use the softmax function since we think of
        the output of the model as probability.
    Note:        
        train or test data must be torch.Size([a, b, c, d])
        size explanation tensor ( [ # N_samples , # 1, hight, width ] )                               
                           
        target or label tensor 
        torch.Size([N_samples, total_number_labels/classes])                                 
    '''

    def __init__(self, nb_hidden):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding =1, padding_mode = 'replicate' ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU()        
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, padding =1, padding_mode = 'replicate' ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU()            
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, padding =1, padding_mode = 'replicate' ),            
            nn.BatchNorm2d(32),
            nn.ReLU()            
            )
                
        self.fc1 = nn.Linear( 32 * 11 * 2 , nb_hidden)
        self.fc2 = nn.Linear( nb_hidden , nb_hidden )
        self.fc3 = nn.Linear( nb_hidden , nb_hidden )
        self.fc4 = nn.Linear( nb_hidden , nb_hidden )
        self.fc5 = nn.Linear( nb_hidden, 3)        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = x.view(-1, 32 * 11 * 2  )
        x = F.relu(self.fc1(x))
        x = self.dropout( x )
        x = F.relu(self.fc2(x))
        x = self.dropout( x )
        x = F.relu(self.fc3(x))
        x = self.dropout( x )        
        x = F.relu(self.fc4(x))        
        x = self.dropout( x )  
        x = self.fc5(x)                 
        return F.softmax(x, dim = 1)
