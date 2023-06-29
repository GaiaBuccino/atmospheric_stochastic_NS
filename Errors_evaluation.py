### Evaluating errors
"""
Module for FNN-Autoencoders.
"""
import pandas as pd
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

#Different test possibilities: discontinuity, snapshots

types_test = ["snapshots"]  #"snapshots" or "discontinuity"

for test in types_test:
    if test == "snapshots":
        
        #DATA LOADING
        print("DEALING WITH SNAPSHOTS...")
        train_FOM = np.load('snap_w_training.npy')
        test_FOM = np.load('snap_w_testing.npy')
        params_training = np.load('params_training.npy')
        params_testing = np.load('params_testing.npy')

        print("data train shape", train_FOM.shape)
        print("data test shape", test_FOM.shape)

        
    elif test == "discontinuity":
        print("DEALING WITH DISCONTINUITY...")

        #DATA LOADING

        train_FOM = np.load('discontinuity_training.npy')
        test_FOM = np.load('discontinuity_testing.npy')
        params_training = np.load('disc_params_training.npy')
        params_testing = np.load('disc_params_testing.npy')

        print("data train FOM shape", train_FOM.shape)
        print("data test FOM shape", test_FOM.shape)


    #setting correct dimensions for convolutional layers (1 input channel)
    train_torch = np.expand_dims(train_FOM, axis=1)
    test_torch = np.expand_dims(test_FOM, axis=1)

    train_POD = train_FOM.reshape(180, 65536)
    test_POD = test_FOM.reshape(20, 65536)

    print("data train POD shape", train_POD.shape)
    print("data test POD shape", test_POD.shape)

    fake_f = torch.nn.ELU
    #optim = torch.optim.Adam
    conv_layers = 6
    epochs = 10000
    fake_val = 2
    neurons_linear = fake_val

    #convolutional neural network fake initialization
    conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, neurons_linear)  #fake structure

    mypath_test = f'./Stochastic_results/{test}_tests' #conv_AE_{epochs}epochs_{conv_layers}_conv_layers_{neurons_linear}_linear_neurons_{test}'
    #creating tests folder
    if not os.path.isdir(mypath_test):
        os.makedirs(mypath_test)

    mypath_conv = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv' #conv_AE_{epochs}epochs_{conv_layers}_conv_layers_{neurons_linear}_linear_neurons_{test}'
    #creating layers folder
    if not os.path.isdir(mypath_conv):
        os.makedirs(mypath_conv)

    mypath_epochs = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs'
    #creating epochs folder
    if not os.path.isdir(mypath_epochs):
        os.makedirs(mypath_epochs)

    neurons_linear = [2,6,14,30]
    #error definition and computation
    which_error = ["convAE-FOM", "POD_NN-encoder", "POD_NN_decoder-convAE"]
    db_training = Database(params_training, train_POD)
    db_testing = Database(params_testing, test_POD)
    ann = ANN([16,64,64], nn.Tanh(), [50000, 1e-12])


    for dim in neurons_linear:
        mypath_neurons = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons'
        #looking for exixting directory or creating a new one
            
        if not os.path.isdir(mypath_neurons):
        
            print("The conv_ae specified does not exist, training...")
            os.makedirs(mypath_neurons)
            conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, dim)  #epochs and neurons_linear are passed to the conv_ae
            start = time()
            conv_ae.fit(train_torch, test_torch) 
            end = time()
            time_train = end-start
            #capire come salvare in un file la variabile time_train 
            torch.save(conv_ae, f'{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt')

        else:

            #cnn real initialization -> fitted and saved in conv_ae_model_construction 

            print("The conv_ae specified already exists, loading...")
            conv_ae = torch.load(f'{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt') 

        """fin qui okkkkk"""

        #reduction-expansion
        r_train = conv_ae.transform(train_torch)
        e_train = conv_ae.inverse_transform(r_train).T.squeeze().reshape(180, 65536)
        r_train = r_train.T
        print("shape e_train ", e_train.shape)
        r_test = conv_ae.transform(test_torch)
        e_test = conv_ae.inverse_transform(r_test).T.squeeze().reshape(20, 65536)
        r_test = r_test.T

        train_err = np.zeros(len(train_FOM))
        test_err = np.zeros(len(test_FOM))

        
        for type_err in which_error:

            if type_err == "convAE-FOM":
                for ii in range(len(train_FOM)):
                
                    train_err[ii] = np.linalg.norm(e_train[ii] - train_POD[ii])/np.linalg.norm(train_POD[ii])

                #plot error TRAINING
                plt.figure()
                plt.semilogy(params_training, train_err, 'bo-')
                plt.title(f"{test}test_training_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                plt.ylabel("relative error")
                plt.xlabel("time")

                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Training_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()
                train_err = np.zeros(len(train_FOM))

                for jj in range(len(test_FOM)):
                
                    test_err[jj] = np.linalg.norm(e_test[jj] - test_POD[jj])/np.linalg.norm(test_POD[jj])

                #plot error TRAINING
                plt.figure()
                plt.semilogy(params_testing, test_err, 'ro-')
                plt.title(f"{test}test_testing_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                plt.ylabel("relative error")
                plt.xlabel("time")

                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Testing_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()
                test_err = np.zeros(len(test_FOM))

            if type_err == "POD_NN-encoder" or "POD_NN_decoder-convAE":

                # ROM model
                pod = POD('svd', rank = dim)
                rom = ROM(db_training, pod, ann)
                start_rom = time()
                rom.fit()
                end_rom = time()

                print("time to train rom = ", end_rom - start_rom)

                rom_r_train = rom.reduction.transform(db_training.snapshots.T).T
                rom_r_test = rom.reduction.transform(db_testing.snapshots.T).T

                if type_err == "POD_NN-encoder":
                    for ii in range(len(train_FOM)):
                    
                        train_err[ii] = np.linalg.norm(r_train[ii] - rom_r_train[ii])/np.linalg.norm(rom_r_train[ii])

                    #plot error TRAINING
                    plt.figure()
                    plt.semilogy(params_training, train_err, 'bo-')
                    plt.title(f"{test}test_training_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Training_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()
                    train_err = np.zeros(len(train_FOM))

                    for jj in range(len(test_FOM)):
                    
                        test_err[jj] = np.linalg.norm(r_test[jj] - rom_r_test[jj])/np.linalg.norm(rom_r_test[jj])

                    #plot error TRAINING
                    plt.figure()
                    plt.semilogy(params_testing, test_err, 'ro-')
                    plt.title(f"{test}test_testing_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Testing_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()
                    test_err = np.zeros(len(test_FOM))

                elif type_err == "POD_NN_decoder-convAE":
                    
                    decoded_rom_train = conv_ae.inverse_transform(rom_r_train.T).T
                    decoded_rom_train = decoded_rom_train.squeeze().reshape(180,65536)
                    decoded_rom_test = conv_ae.inverse_transform(rom_r_test.T).T
                    decoded_rom_test = decoded_rom_test.squeeze().reshape(20,65536)

                    for ii in range(len(train_FOM)):
                        
                        train_err[ii] = np.linalg.norm(e_train[ii] - decoded_rom_train[ii])/np.linalg.norm(decoded_rom_train[ii])

                    #plot error TRAINING
                    plt.figure()
                    plt.semilogy(params_training, train_err, 'bo-')
                    plt.title(f"{test}test_training_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Training_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()
                    train_err = np.zeros(len(train_FOM))

                    for jj in range(len(test_FOM)):
                    
                        test_err[jj] = np.linalg.norm(e_test[jj] - decoded_rom_test[jj])/np.linalg.norm(decoded_rom_test[jj])

                    #plot error TESTING
                    plt.figure()
                    plt.semilogy(params_testing, test_err, 'ro-')
                    plt.title(f"{test}test_testing_error_{type_err}_{conv_layers}conv_layers_{epochs}ep")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Testing_error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()
                    test_err = np.zeros(len(test_FOM))









