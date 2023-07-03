### Evaluating errors
"""
Module for FNN-Autoencoders
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
#from ezyrb.conv_ae import convAE
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

        data_FOM = np.load('snap_w.npy')
        params_FOM = np.load('params.npy')
        train_FOM = np.load('snap_w_training.npy')
        test_FOM = np.load('snap_w_testing.npy')
        params_training = np.load('params_training.npy')
        params_testing = np.load('params_testing.npy')

        print("data train shape", train_FOM.shape)
        print("data test shape", test_FOM.shape)

        #print("params_training", params_training)
        #print("params_testing", params_testing)


        
    elif test == "discontinuity":
        print("DEALING WITH DISCONTINUITY...")

        #DATA LOADING
        
        data_FOM = np.load('discontinuity_snapshots.npy')
        params_FOM = np.load('disc_params.npy')
        train_FOM = np.load('discontinuity_training.npy')
        test_FOM = np.load('discontinuity_testing.npy')
        params_training = np.load('disc_params_training.npy')
        params_testing = np.load('disc_params_testing.npy')

        print("data train FOM shape", train_FOM.shape)
        print("data test FOM shape", test_FOM.shape)

        #print("params_training", params_training)
        #print("params_testing", params_testing)



    #setting correct dimensions for convolutional layers (1 input channel)
    train_torch = np.expand_dims(train_FOM, axis=1)
    test_torch = np.expand_dims(test_FOM, axis=1)

    train_POD = train_FOM.reshape(180, 65536)
    test_POD = test_FOM.reshape(20, 65536)

    print("data train POD shape", train_POD.shape)  #180, 65536
    print("data test POD shape", test_POD.shape)    #20, 65536

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
    which_error = ["convAE-FOM", "POD-inv_transform", "POD_NN-inv_transform"]
    db_data = Database(params_FOM, data_FOM)
    db_training = Database(params_training, train_POD)
    db_testing = Database(params_testing, test_POD)
    ann = ANN([16,64,64], nn.Tanh(), [50000, 1e-12])


    for dim in neurons_linear:
        mypath_neurons = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons'
        #looking for exixting directory or creating a new one
            
        if not (os.path.isdir(mypath_neurons) and os.path.isfile(f"{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt")):
        
            print("The conv_ae specified does not exist, training...")
            
            if not os.path.isdir(mypath_neurons):
                
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

        

        #reduction-expansion
        r_train = conv_ae.transform(train_torch)
        e_train = conv_ae.inverse_transform(r_train).T.squeeze()#.reshape(180, 65536)
        r_train = r_train.T
        print("shape e_train ", e_train.shape)
        r_test = conv_ae.transform(test_torch)
        e_test = conv_ae.inverse_transform(r_test).T.squeeze()#.reshape(20, 65536)
        r_test = r_test.T

        train_err_conv = np.zeros(len(train_FOM))
        test_err_conv = np.zeros(len(test_FOM))

        
        for type_err in which_error:

            if type_err == "convAE-FOM":
                for ii in range(len(train_FOM)):
                
                    train_err_conv[ii] = np.linalg.norm(e_train[ii] - train_FOM[ii])/np.linalg.norm(train_FOM[ii])

                for jj in range(len(test_FOM)):
                
                    test_err_conv[jj] = np.linalg.norm(e_test[jj] - test_FOM[jj])/np.linalg.norm(test_FOM[jj])

                #plot error TRAINING
                plt.figure()
                plt.subplot(1,2,1)
                plt.semilogy(params_training, train_err_conv, 'bo-')
                plt.title(f"{type_err} training error")
                plt.ylabel("relative error")
                plt.xlabel("time")

                plt.subplot(1,2,2)
                plt.semilogy(params_testing, test_err_conv, 'ro-')
                plt.title(f"{type_err} testing error")
                #plt.ylabel("relative error")
                plt.xlabel("time")

                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()
                
                plt.figure()
                plt.semilogy(params_training, train_err_conv, 'bo-')
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_training.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()

                plt.figure()
                plt.semilogy(params_testing, test_err_conv, 'ro-')
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_testing.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()

            if type_err == "POD-inv_transform" or "POD_NN-inv_transform":
                
                # POD model
                pod = POD('svd', rank = dim)
                pod.fit(db_training.snapshots)
                pod_r_train = pod.transform(db_training.snapshots)
                print("shape rom_r_train", pod_r_train.shape)
                pod_e_train = pod.inverse_transform(pod_r_train)

                train_err_POD = np.zeros(len(train_POD))
                #test_err_POD = np.zeros(len(test_POD))
                if type_err == "POD-inv_transform":

                    for ii in range(len(train_FOM)):
                    
                        train_err_POD[ii] = np.linalg.norm(train_POD[ii] - pod_e_train[ii])/np.linalg.norm(train_POD[ii])

                    
                    # for jj in range(len(test_FOM)):
                    
                    #     test_err_POD[jj] = np.linalg.norm(test_POD[jj] - pod_e_test[jj])/np.linalg.norm(test_POD[jj])


                    #plot error TRAINING
                    plt.figure(figsize=(15, 15))
                    plt.subplot(1,2,1)
                    plt.semilogy(params_training, train_err_POD, 'bo-')
                    plt.title(f"{type_err}_error_Train")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.subplot(1,2,2)
                    plt.semilogy(params_training, train_err_conv, 'bo-')
                    plt.title("convAE_error_Train")
                    plt.ylabel("relative error")
                    plt.xlabel("time")
                    
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    #single plot
                    
                    plt.figure()
                    plt.semilogy(params_training, train_err_POD, 'bo-')
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_training.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    
                elif type_err == "POD_NN-inv_transform":

                    rom = ROM(db_training, pod, ann)
                    start_rom = time()
                    rom.fit()
                    end_rom = time()
                    print("time to train rom = ", end_rom - start_rom)

                    train_err_POD_NN = np.zeros(len(train_POD))
                    test_err_POD_NN = np.zeros(len(test_POD))
                    
                    r_rom_train = rom.predict(params_training)
                    #r_rom_train = rom.reduction.transform(db_training.parameters.T)
                    #e_rom_train = rom.reduction.inverse_transform(r_rom_train.T).T
                    #e_rom_train = decoded_rom_train.squeeze().reshape(180,65536)
                    r_rom_test = rom.predict(params_testing)
                    #e_rom_test = rom.reduction.inverse_transform(r_rom_test).T

                    
                    for ii in range(len(train_FOM)):
                    
                        train_err_POD_NN[ii] = np.linalg.norm(train_POD[ii] - r_rom_train[ii])/np.linalg.norm(train_POD[ii])

                    for jj in range(len(test_FOM)):
                    
                        test_err_POD_NN[jj] = np.linalg.norm(test_POD[jj] - r_rom_test[jj])/np.linalg.norm(test_POD[jj])

                    
                    #plot error TRAINING
                    plt.figure(figsize=(15, 15))
                    plt.subplot(2,2,1)
                    plt.semilogy(params_training, train_err_POD_NN, 'bo-')
                    plt.title(f"{type_err}_error_Train")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.subplot(2,2,2)
                    plt.semilogy(params_training, train_err_conv, 'bo-')
                    plt.title("convAE_error_Train")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    #plot error TESTING
                    plt.subplot(2,2,3)
                    plt.semilogy(params_testing, test_err_POD_NN, 'ro-')
                    plt.title(f"{type_err}_error_Test")
                    plt.ylabel("relative error")
                    plt.xlabel("time")

                    plt.subplot(2,2,4)
                    plt.semilogy(params_testing, test_err_conv, 'ro-')
                    plt.title("convAE_error_Test")
                    plt.ylabel("relative error")
                    plt.xlabel("time")
                    
                    
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()


                    plt.figure()
                    plt.semilogy(params_training, train_err_POD_NN, 'bo-')
                    #plt.title(f"{type_err}_error_Train")
                    #plt.ylabel("relative error")
                    #plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_training.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    plt.figure()
                    plt.semilogy(params_testing, test_err_POD_NN, 'ro-')
                    #plt.title(f"{type_err}_error_Train")
                    #plt.ylabel("relative error")
                    #plt.xlabel("time")

                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_testing.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()


