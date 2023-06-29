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

    
""" train_data = torch.from_numpy(np.load(f"snap_w_training.npy"))
train_params = torch.from_numpy(np.load("params_training.npy"))
test_data = torch.from_numpy(np.load(f"snap_w_testing.npy"))
test_params = torch.from_numpy(np.load("params_testing.npy"))
 """

""" data_dir = 'dataset'
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True) """

my_path = os.path.abspath(__file__)

train_dataset = np.load('snap_w_training.npy')
test_dataset = np.load('snap_w_testing.npy')
params_training = np.load('params_training.npy')
params_testing = np.load('params_testing.npy')
#train_data = train_dataset.data.numpy()
train_data = np.expand_dims(train_dataset, axis=1)
test_data = np.expand_dims(test_dataset, axis=1)

""" mask_train = [True, True, True,True, True, True,True, True, True, False ]*20
mask_train = np.array(mask_train)
mask_test = np.array(~mask_train)

print(f"snap_testing: {sum(mask_test)}")

params_training = params_training[mask_train]
snap_training = train_dataset[mask_train, :]
snapshots_training = snap_training.reshape(180,256*256)
params_testing = params_testing[mask_test]
snap_testing = test_dataset[mask_test, :]
snapshots_testing = snap_testing.reshape(20,256*256)
#train_data = train_dataset.data.numpy()
train_data = np.expand_dims(snap_training, axis=1)
test_data = np.expand_dims(snap_testing, axis=1) """


print("test_data_shape", test_data.shape)
print("params_training_shape", params_training.shape)

f = torch.nn.ELU
low_dim = 5
optim = torch.optim.Adam
epochs = 10000

conv_ae = convAE([14], [14], f(), f(), epochs)  #fake structure
conv_ae = torch.load(f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers.pt')

snap_training = train_data.reshape(180, 256*256)
snap_testing = test_data.reshape(20, 256*256)

db_training = Database(params_training, snap_training)
db_testing = Database(params_testing, snap_testing) 

ann = ANN([16, 64, 64], nn.Tanh(), [50000, 1e-12])
pod = POD('svd', rank=14)

rom = ROM(db_training, pod, ann)
#rom.load("discontinuity_nn.rom")
rom.fit()
rom.save("discontinuity_nn.rom")
rom_reduced_train = rom.reduction.transform(snap_training.T)
rom_reduced_train = rom_reduced_train.T
rom_reduced_test = rom.reduction.transform(snap_testing.T)
rom_reduced_test = rom_reduced_test.T
print("shape rom reduced training", rom_reduced_train.shape)
#ann.fit(params_training, snap_training)

#training reduction-expansion
reduced_train_snapshots = conv_ae.transform(train_data)
expanded_train_snapshots = conv_ae.inverse_transform(reduced_train_snapshots)
reduced_train_snapshots = reduced_train_snapshots.T
e_train_snapshots = expanded_train_snapshots.T.squeeze()
model_training_err = np.zeros(len(e_train_snapshots))

#testing reduction-expansion
reduced_test_snapshots = conv_ae.transform(test_data)
expanded_test_snapshots = conv_ae.inverse_transform(reduced_test_snapshots)
reduced_test_snapshots = reduced_test_snapshots.T
e_test_snapshots = expanded_test_snapshots.T.squeeze()
model_testing_err = np.zeros(len(e_test_snapshots))
print("shape reduced test snapshots", reduced_test_snapshots.shape)

for i in range(len(e_test_snapshots)):
   
    model_testing_err[i] = np.linalg.norm(rom_reduced_test[i] - reduced_test_snapshots[i])/np.linalg.norm(rom_reduced_test[i])

for i in range(len(e_train_snapshots)):
   
    model_training_err[i] = np.linalg.norm(rom_reduced_train[i] - reduced_train_snapshots[i])/np.linalg.norm(rom_reduced_train[i])
    

#plot error TESTING
plt.figure()
plt.semilogy(params_testing, model_testing_err, 'ro-')
plt.title(f"Reconstruction of the Testing POD-NN with convAE_{epochs}_encoder epochs")
plt.ylabel("w relative error")
plt.xlabel("time")

plt.savefig(f'./Stochastic_results/Reconstruction_Testing_FOM_convAE{epochs}epochs_6_conv_layers_CHECK.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
plt.close()

#plot error TRAINING
plt.figure()
plt.semilogy(params_training, model_training_err, 'bo-')
plt.title(f"Reconstruction of the Training POD-NN with convAE_{epochs}_encoder epochs")
plt.ylabel("w relative error")
plt.xlabel("time")

plt.savefig(f'./Stochastic_results/Reconstruction_Training_FOM_convAE{epochs}epochs_encoder_6_conv_layers_CHECK.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
plt.close()
