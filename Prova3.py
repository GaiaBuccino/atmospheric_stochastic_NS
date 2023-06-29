### Evaluating errors
"""
Module for FNN-Autoencoders.
"""
import conv_ae
from conv_ae import convAE
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
from ezyrb.reduction import Reduction
from ezyrb.ann import ANN
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, GPR, ANN, AE
from ezyrb import ReducedOrderModel as ROM
import pickle
import copy
import os
from time import time

    

my_path = os.path.abspath(__file__)

train_dataset = np.load('snap_w_training.npy')
test_dataset = np.load('snap_w_testing.npy')
params_training = np.load('params_training.npy')
params_testing = np.load('params_testing.npy')
#train_data = train_dataset.data.numpy()
train_data = np.expand_dims(train_dataset, axis=1)
test_data = np.expand_dims(test_dataset, axis=1)

print("test_data_shape", test_data.shape)

f = torch.nn.ELU
low_dim = 5
optim = torch.optim.Adam
epochs = 10000
neurons_linear = 2

conv_ae = convAE([14], [14], f(), f(), epochs, neurons_linear)

#Saving
start = time()
conv_ae.fit(train_data, test_data) 
end = time() 

print("time required for the training", end-start)

torch.save(conv_ae, f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers.pt')

#conv_ae = torch.load(f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers.pt')

#training reduction-expansion
reduced_train_snapshots = conv_ae.reduce(train_data)
expanded_train_snapshots = conv_ae.expand(reduced_train_snapshots)

#testing reduction-expansion
reduced_test_snapshots = conv_ae.reduce(test_data)
expanded_test_snapshots = conv_ae.expand(reduced_test_snapshots)

#print("ciao")
e_test_snapshots = expanded_test_snapshots.T.squeeze()
model_testing_err = np.zeros(len(e_test_snapshots))

e_train_snapshots = expanded_train_snapshots.T.squeeze()
model_training_err = np.zeros(len(e_train_snapshots))

for i in range(len(e_test_snapshots)):
   
    model_testing_err[i] = np.linalg.norm(e_test_snapshots[i] - test_dataset[i][:][:])/np.linalg.norm(test_dataset[i][:][:])

for i in range(len(e_train_snapshots)):
   
    model_training_err[i] = np.linalg.norm(e_train_snapshots[i] - train_dataset[i][:][:])/np.linalg.norm(train_dataset[i][:][:])
    

#plot error TESTING
plt.figure()
plt.semilogy(params_testing, model_testing_err, 'ro-')
plt.title(f"Reconstruction of the Testing FOM snapshot with convAE_{epochs} epochs")
plt.ylabel("w relative error")
plt.xlabel("time")

plt.savefig(f'./Stochastic_results/Reconstruction_Testing_convAE{epochs}epochs-FOM_6_conv_layers.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
plt.close()

#plot error TRAINING
plt.figure()
plt.semilogy(params_training, model_training_err, 'ro-')
plt.title(f"Reconstruction of the Training FOM snapshot with convAE_{epochs} epochs")
plt.ylabel("w relative error")
plt.xlabel("time")

plt.savefig(f'./Stochastic_results/Reconstruction_Training_convAE{epochs}epochs-FOM_6_conv_layers.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
plt.close()



 