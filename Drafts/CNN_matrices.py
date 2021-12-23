#!/usr/bin/env python

# data packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

# torch libraries

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

# import dlc_practical_prologue as prologue
# train_input, train_target, test_input, test_target = \
#     prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

##########################################################
### data splitting
def split_data(x, y, ratio=0.90, seed=0):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    dataset_size = x.shape[0]
    indices = np.random.permutation(dataset_size)
    threshold  = int(ratio * dataset_size)
    index_train = indices[:threshold]
    index_test = indices[threshold:]
    # create split
    x_training = x[index_train]
    x_test = x[index_test]
    y_training = y[index_train]
    y_test = y[index_test]
    return x_training, x_test, y_training, y_test

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
print(len(df_filter.index) + discard_data - len(df.index))

##########################################################
# separation into experiments
# contruction of labels to numerical values

total_samples = 0
counter = 0 
window_size = 20
num_pulses=int(max(df_filter['pulse'].values)) # tot number of different pulses == tot number of different experiments

number_correct_samples = 127832
all_samples  = np.zeros(127832 * 20 * 4).reshape((-1, 20, 4))
all_labels = np.zeros(127832 * 3).reshape((-1, 3))

for k in range(num_pulses):    
    print('running experiment ', k+1 )
    mask_experiment = df_filter.pulse == k + 1
    df_experiment = df_filter[mask_experiment]
    df_experiment = df_experiment.reset_index(drop=True)    
    
    # labels
    maskl = df_experiment.LDH == 'L'
    maskd = df_experiment.LDH == 'D'
    maskh = df_experiment.LDH == 'H'
    labels = np.vstack((maskl, maskd, maskh)).T + 0.0
    
    features_exp = df_experiment.keys().to_numpy()
    mask_features_exp = np.array([False, True, True, True, True, False, False ])
    features_exp = features_exp[mask_features_exp]
    x_exp = np.array( df_experiment.loc[:, features_exp].values )    

    # this number varies from one experiment to another  
    num_samples = int( x_exp.shape[0]/window_size )
    step = 0
    
    for i in range(num_samples):
        all_samples[i + counter] = x_exp[ step : step + window_size, : ].reshape((-1, 20, 4)) 
        all_labels[i + counter] = labels[ step : step + window_size, : ].mean(axis = 0).reshape((-1, 1, 3))                
        step += window_size
        
    counter +=num_samples
    total_samples  += num_samples
    
# create of train and test set
validation_split = 0.50
shuffle_dataset = True
random_seed= 0
dataset_size = all_samples.shape[0]
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

x_train = all_samples[train_indices]
y_train = all_labels[train_indices]

x_test = all_samples[test_indices]
y_test = all_labels[test_indices]

# transform into a tensor
x_train = torch.from_numpy(x_train).float().reshape(-1, 1, 20, 4)[: 63900 ]
y_train = torch.from_numpy(y_train).float()[: 63900 ]
x_test = torch.from_numpy(x_test).float().reshape(-1, 1, 20, 4)[: 63900 ]
y_test = torch.from_numpy(y_test).float()[: 63900 ]  

print('train set shape: ', x_train.shape)
print('test set shape: ', x_test.shape)

### info regarding data shape for models to work 
'''
    train or test data must be torch.Size([1000, 1, 28, 28])
    size explanation tensor ( [ # N_samples , # channels_input, hight, width ] )                               
                       
    target or label tensor 
    torch.Size([N_samples, # labels/classes])
                                   
'''

#%%
######################################################################
class Net2(nn.Module):
    def __init__(self, nb_hidden):        
        super().__init__()        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding =1, padding_mode = 'replicate' )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, padding =1, padding_mode = 'replicate' )
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, padding =1, padding_mode = 'replicate' )
        self.fc1 = nn.Linear( 64 * 6 * 2 , nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 3)
        self.bn1 = nn.BatchNorm2d(32) # batch normalization
        self.bn3 = nn.BatchNorm2d(64) # batch normalization

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)
        x = self.bn1(x) 
        x = torch.relu(x)        
        
        x = F.max_pool2d(self.conv2(x), kernel_size=2)
        x = self.bn1(x)     
        x = torch.relu(x)        
        
        x = torch.relu(self.conv3(x))
        x = self.bn3(x)     
        x = torch.relu(self.fc1(x.view(-1, 768)))
        
        x = self.fc2(x) 
        x = torch.softmax(x, dim = 0)
        return x
    
######################################################################

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs = 10):
    lr = 1e-8
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)    

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):            
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    eps = 0.2 # tolerance
    nb_data_errors = 0    
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))    
        #predicted_classes , _ = output.max(dim  = 0)     
        for k in range(mini_batch_size):
            if torch.norm(data_target[b + k] - output[k])  > eps :
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors            

# def compute_nb_errors(model, data_input, target, mini_batch_size):
#     nb_errors = 0
#     for b in range(0, data_input.size(0), mini_batch_size):
#         output = model(data_input.narrow(0, b, mini_batch_size))
#         _, predicted_classes = output.max(1)
#         for k in range(mini_batch_size):
#             if target[b + k, predicted_classes[k]] <= 0:
#                 nb_errors = nb_errors + 1
#     return nb_errors

#######################################################################
mini_batch_size = 100
nb_hidden = 10
# fake data to test if model runs
# torch.manual_seed(100)

# to train in GPU if available 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda')
# print('working device ', device)
# Net2.to(device)
# x_test, y_test = x_test.to(device), y_test.to(device)

model = Net2(nb_hidden)
model.train()
print('training started')
train_model(model, x_train, y_train, mini_batch_size)
print('training finished')

#%%
#######################################################################
def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    eps = 0.2 # tolerance
    nb_data_errors = 0    
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))    
        #predicted_classes , _ = output.max(dim  = 0)     
        print(output)
    return nb_data_errors            

nb_test_errors = compute_nb_errors(model, x_test, y_test, mini_batch_size)
print('test error Net2 {:0.2f}%% {:d}/{:d}'.format((100 * nb_test_errors) / x_train.size(0),
                                                    nb_test_errors, x_test.size(0)))
# test functions
# x_test.narrow(0, 0, mini_batch_size).shape