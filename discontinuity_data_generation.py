#Evaluating discontinuity errors

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

tests = ["synthetic_discontinuity_stats"]

x0_lim = 0
y0_lim = 0
t0_lim = 0.3
x1_lim =1
y1_lim = 1
t1_lim = 3
x_dim =256
y_dim = 256
t_dim = 0
t_train=0
t_test=0
mask_train=[]
mask_test=[]

for test in tests:

    if 'stats' not in f'{test}':
       
        print(f"Generating {test} data")
        t_dim = 200
        t_train = 180
        t_test = 20
        mask_train = [True, True, True,True, True, True,True, True, True,False]*20
        mask_train = np.array(mask_train)
        mask_test = np.array(~mask_train)

    else:

        print(f"Dealing with {test} data")
        t_dim = 1001
        t_train = 1000
        t_test = 1
        mask_train = [True, True, True,True, True, True,True, True, True,True]*100 + [False]
        mask_train = np.array(mask_train)
        mask_test = np.array(~mask_train)



    xx = np.linspace(x0_lim,x1_lim,x_dim)
    yy = np.linspace(y0_lim,y1_lim,y_dim)
    tt = np.linspace(t0_lim,t1_lim,t_dim)
    tt =np.expand_dims(tt, axis=1)
    zz = np.zeros((tt.shape[0],xx.shape[0],yy.shape[0]))
    f = lambda t,x,y : (1.+0.1*np.sin(2*np.pi*(x-0.1*t))*np.cos(2*np.pi*y))*(x>t*y)+(x<=t*y)*(1.5+0.2*np.cos(6.*np.pi*x)*np.sin(2*np.pi*(y-0.1*t)))


    for it, t in enumerate(tt):
        for ix, x in enumerate(xx):
            zz[it,ix,:] = f(t,x,yy)



    print(f"snap_training: {sum(mask_train)}")
    print(f"snap_testing: {sum(mask_test)}")

    disc_snapshots = zz.reshape(t_dim, x_dim*y_dim)
    disc_params = tt

    params_training = tt[mask_train]
    snap_training = zz[mask_train, :]
    snapshots_training = snap_training.reshape(t_train,x_dim*y_dim)
    params_testing = tt[mask_test]
    snap_testing = zz[mask_test, :]
    snapshots_testing = snap_testing.reshape(t_test,x_dim*y_dim)
    #train_data = train_dataset.data.numpy()
    np.save(f"Data/{test}_training.npy", snap_training)
    np.save(f"Data/{test}_testing.npy", snap_testing)
    np.save(f"Data/{test}_params_training.npy", params_training)
    np.save(f"Data/{test}_params_testing.npy", params_testing)

# np.save("discontinuity_snapshots.npy", disc_snapshots)
# np.save("disc_params.npy", disc_params)

# print("disc_data_shape", disc_snapshots.shape)
# print("disc_params_shape", disc_params.shape)


# train_data = np.expand_dims(snap_training, axis=1)
# test_data = np.expand_dims(snap_testing, axis=1)

# print("snap_training shape", snap_training.shape)

# print("test_data_shape", test_data.shape)
# print("params_training_shape", params_training.shape)

# # f = torch.nn.ELU
# # low_dim = 5
# # optim = torch.optim.Adam
# # epochs = 10000

# # conv_ae = convAE([14], [14], f(), f(), epochs)  #fake structure
# # conv_ae = torch.load(f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers_DISCONTINUITY.pt')

# # db_training = Database(params_training, snapshots_training)
# # db_testing = Database(params_testing, snap_testing) 

# # ann = ANN([16, 64, 64], nn.Tanh(), [50000, 1e-12])
# # pod = POD('svd', rank=14)

# # rom = ROM(db_training, pod, ann)
# # #rom.load("discontinuity_nn.rom")
# # rom.fit()
# # rom.save("discontinuity_nn.rom")
# # rom_reduced_train = rom.reduction.transform(snapshots_training.T)
# rom_reduced_train = rom_reduced_train.T
# rom_reduced_test = rom.reduction.transform(snapshots_testing.T)
# rom_reduced_test = rom_reduced_test.T
# print("shape rom reduced training", rom_reduced_train.shape)
# #ann.fit(params_training, snap_training)

# #training reduction-expansion
# reduced_train_snapshots = conv_ae.transform(train_data)
# expanded_train_snapshots = conv_ae.inverse_transform(reduced_train_snapshots)
# reduced_train_snapshots = reduced_train_snapshots.T
# e_train_snapshots = expanded_train_snapshots.T.squeeze()
# model_training_err = np.zeros(len(e_train_snapshots))

# #testing reduction-expansion
# reduced_test_snapshots = (conv_ae.transform(test_data))
# expanded_test_snapshots = conv_ae.inverse_transform(reduced_test_snapshots)
# reduced_test_snapshots = reduced_test_snapshots.T
# e_test_snapshots = expanded_test_snapshots.T.squeeze()
# model_testing_err = np.zeros(len(e_test_snapshots))
# print("shape reduced test snapshots", reduced_test_snapshots.shape)

# for i in range(len(e_test_snapshots)):
   
#     model_testing_err[i] = np.linalg.norm(reduced_test_snapshots[i] - rom_reduced_test[i])/np.linalg.norm(rom_reduced_test[i])

# for i in range(len(e_train_snapshots)):
   
#     model_training_err[i] = np.linalg.norm(reduced_train_snapshots[i] - rom_reduced_train[i])/np.linalg.norm(rom_reduced_train[i])
    

# #plot error TESTING
# plt.figure()
# plt.semilogy(params_testing, model_testing_err, 'ro-')
# plt.title(f"Reconstruction of the Testing POD-NN with convAE_{epochs}_encoder epochs")
# plt.ylabel("w relative error")
# plt.xlabel("time")

# plt.savefig(f'./Stochastic_results/Reconstruction_Testing_POD-NN_convAE{epochs}epochs_encoder_6_conv_layers_DISCONTINUITY_time-dep.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
# plt.close()

# #plot error TRAINING
# plt.figure()
# plt.semilogy(params_training, model_training_err, 'bo-')
# plt.title(f"Reconstruction of the Training POD-NN with convAE_{epochs}_encoder epochs")
# plt.ylabel("w relative error")
# plt.xlabel("time")

# plt.savefig(f'./Stochastic_results/Reconstruction_Training_POD-NN_convAE{epochs}epochs_encoder_6_conv_layers_DISCONTINUITY_time-dep.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
# plt.close()