### Evaluating errors
"""
Module for FNN-Autoencoders

"""
import pandas as pd
import conv_ae
from conv_ae import convAE
import torch
from torch import nn
# import torchvision
# from torchvision import datasets, transforms
import numpy as np
from ezyrb.reduction import Reduction
from ezyrb.ann import ANN
#from ezyrb.conv_ae import convAE
# import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, GPR, ANN, AE
from ezyrb import ReducedOrderModel as ROM
import pickle
import copy
import os
from time import time

#Different test possibilities: discontinuity, snapshots

types_test = ["discontinuity"]  #"snapshots" or "discontinuity"

for test in types_test:
    if test == "snapshots":
        
        #DATA LOADING
        print("DEALING WITH SNAPSHOTS...")

        # data_FOM = np.load('snap_w.npy')
        # params_FOM = np.load('params.npy')
        train_FOM = np.load('snap_w_training.npy')
        test_FOM = np.load('snap_w_testing.npy')
        params_training = np.load('params_training.npy')
        params_testing = np.load('params_testing.npy')

        print("data train shape", train_FOM.shape)
        print("data test shape", test_FOM.shape)

        print("params_training", params_training.shape)
        print("params_testing", params_testing.shape)


        
    elif test == "discontinuity":
        print("DEALING WITH DISCONTINUITY...")

        #DATA LOADING
        
        #data_FOM = np.load('discontinuity_snapshots.npy')
        #params_FOM = np.load('disc_params.npy')
        train_FOM = np.load('discontinuity_training.npy')
        test_FOM = np.load('discontinuity_testing.npy')
        params_training = np.load('disc_params_training.npy')
        params_testing = np.load('disc_params_testing.npy')

        print("data train FOM shape", train_FOM.shape)  #(180, 256, 256)
        print("data test FOM shape", test_FOM.shape)

        #print("params_training", params_training)
        #print("params_testing", params_testing)

    

    #setting correct dimensions for convolutional layers (1 input channel)
    train_torch = np.expand_dims(train_FOM, axis=1)
    test_torch = np.expand_dims(test_FOM, axis=1)

    print("data train torch shape", train_torch.shape)  #180, 65536
    print("data test torch shape", test_torch.shape) 

    train_POD = train_FOM.reshape(180, 65536)
    test_POD = test_FOM.reshape(20, 65536)

    print("data train POD shape", train_POD.shape)  #180, 65536
    print("data test POD shape", test_POD.shape)    #20, 65536



    fake_f = torch.nn.ELU
    #optim = torch.optim.Adam
    conv_layers = 6
    epochs = 8000
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

    neurons_linear = [14,30]    # 2, 6, 14, 30
    #error definition and computation
    which_error = ["convAE-FOM", "POD-inv_transform", "POD_NN-inv_transform", "NN_encoder-FOM"]  
    #db_data = Database(params_FOM, data_FOM)
    db_training = Database(params_training, train_POD)  #train_POD = 180, 65536
    db_testing = Database(params_testing, test_POD)     #test_POD = 20, 65536
    


    for dim in neurons_linear:

        mypath_neurons = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons'
        #looking for exixting directory or creating a new one
            
        if not (os.path.isdir(mypath_neurons) and os.path.isfile(f"{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt")):
        
            print("The conv_ae specified does not exist, training...")
            
            if not os.path.isdir(mypath_neurons):
                
                os.makedirs(mypath_neurons)

            conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, dim)  #epochs and neurons_linear are passed to the conv_ae
            start = time()
            conv_ae.fit(train_torch, test_torch, dim) 
            end = time()
            time_train = end-start
            #capire come salvare in un file la variabile time_train 
            torch.save(conv_ae, f'{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt')

        else:

            #cnn real initialization -> fitted and saved in conv_ae_model_construction 

            print("The conv_ae specified already exists, loading...")
            conv_ae = torch.load(f'{mypath_neurons}/model_{test}_{conv_layers}conv_{epochs}ep_{dim}lin_neurons.pt') 

        plt.figure()

        plt.imshow(train_FOM[-1],cmap=plt.cm.jet,origin='lower')
        plt.title(f'Reconstruction FOM')
        plt.colorbar()
        plt.clim(0.9, 1.7)
        plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/FOM_train_last_solution.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
        plt.close()
        plt.figure()

        plt.imshow(test_FOM[-1],cmap=plt.cm.jet,origin='lower')
        plt.title(f'ReconstructionFOM')
        plt.colorbar()
        plt.clim(0.9, 1.7)
        plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/FOM_test_last_solution.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
        plt.close()
                

        ann_POD = ANN([16,64,64], nn.Tanh(), [60000, 1e-12])     #da correggere in 60000
        ann_enc = ANN([16,64,64], nn.Tanh(), [60000, 1e-12])

        #reduction-expansion
        r_train = conv_ae.transform(train_torch)        #Encoder    
        e_train = conv_ae.inverse_transform(r_train).T.squeeze()#.reshape(180, 65536)       #Decoder
        r_train = r_train.T
        print("shape e_train ", e_train.shape)
        r_test = conv_ae.transform(test_torch)
        e_test = conv_ae.inverse_transform(r_test).T.squeeze()#.reshape(20, 65536)
        r_test = r_test.T 

        train_err_conv = np.zeros(len(train_FOM))
        test_err_conv = np.zeros(len(test_FOM))
        train_err_POD = np.zeros(len(train_POD))
        train_err_POD_NN = np.zeros(len(train_POD))
        test_err_POD_NN = np.zeros(len(test_POD))
        train_err_NN_encoder = np.zeros(len(train_POD))
        test_err_NN_encoder = np.zeros(len(test_POD))

        
        for type_err in which_error:

            if type_err == "convAE-FOM":
                for ii in range(len(train_FOM)):
                
                    train_err_conv[ii] = np.linalg.norm(e_train[ii] - train_FOM[ii])/np.linalg.norm(train_FOM[ii])  #conv_AE reconstructed VS FOM

                for jj in range(len(test_FOM)):
                
                    test_err_conv[jj] = np.linalg.norm(e_test[jj] - test_FOM[jj])/np.linalg.norm(test_FOM[jj])

                
                real_convae_train = e_train[-1].reshape((256, 256))
                real_convae_test = e_test[-1].reshape((256, 256))

                plt.figure()

                plt.imshow(real_convae_train,cmap=plt.cm.jet,origin='lower')
                plt.title(f'Reconstruction{type_err}')
                plt.colorbar()
                plt.clim(0.9, 1.7)
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_train_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()

                plt.figure()

                plt.imshow(real_convae_test,cmap=plt.cm.jet,origin='lower')
                plt.title(f'Reconstruction{type_err}')
                plt.colorbar()
                plt.clim(0.9, 1.7)
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_test_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()

                # #plot error TRAINING
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.semilogy(params_training, train_err_conv, 'bx-')
                # plt.title(f"{type_err} training error")
                # #plt.ylabel("relative error")
                # plt.xlabel("time")
                
                # #plot error TESTING
                # plt.subplot(1,2,2)
                # plt.semilogy(params_testing, test_err_conv, 'ro-')
                # plt.title(f"{type_err} testing error")
                # #plt.ylabel("relative error")
                # plt.xlabel("time")
                # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                # plt.close()
                
                # #plot comparison error TRAINING - TESTING
                # plt.figure()
                # plt.semilogy(params_training, train_err_conv, 'bx-')
                # plt.semilogy(params_testing, test_err_conv, 'ro-')
                # plt.legend(['Train', 'Test'])
                # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                # plt.close()

                """ plt.figure()
                plt.semilogy(params_testing, test_err_conv, 'ro-')
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_testing.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close() """

            if type_err == "POD-inv_transform" or "POD_NN-inv_transform":
                
                # POD model
                pod = POD('svd', rank = dim)
                pod.fit(db_training.snapshots)
                pod_r_train = pod.transform(db_training.snapshots)
                print("shape rom_r_train", pod_r_train.shape)
                pod_e_train = pod.inverse_transform(pod_r_train)

                
                #test_err_POD = np.zeros(len(test_POD))

                if type_err == "POD-inv_transform":

                    for ii in range(len(train_FOM)):
                    
                        train_err_POD[ii] = np.linalg.norm(train_POD[ii] - pod_e_train[ii])/np.linalg.norm(train_POD[ii])     # POD reconstructed VS FOM

                    
                    real_pod_train = pod_e_train[-1].reshape((256, 256))
                    

                    plt.figure()

                    plt.imshow(real_pod_train,cmap=plt.cm.jet,origin='lower')
                    plt.title(f'Reconstruction{type_err}')
                    plt.colorbar()
                    plt.clim(0.9, 1.7)
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_train_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    
                    # for jj in range(len(test_FOM)):
                    
                    #     test_err_POD[jj] = np.linalg.norm(test_POD[jj] - pod_e_test[jj])/np.linalg.norm(test_POD[jj])


                    #plot error TRAINING
                    # POD
                    # plt.figure(figsize=(15, 15))
                    # plt.subplot(1,2,1)
                    # plt.semilogy(params_training, train_err_POD, 'bx-')
                    # plt.title(f"{type_err}_error_Train")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time")

                    # # convAE
                    # plt.subplot(1,2,2)
                    # plt.semilogy(params_training, train_err_conv, 'bo-')
                    # plt.title("convAE_error_Train")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time")
                    
                    # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    # plt.close()

                    # #single plot
                    # # POD error
                    # plt.figure()
                    # plt.semilogy(params_training, train_err_POD, 'bx-')
                    # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_training.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    # plt.close()

                    
                elif type_err == "POD_NN-inv_transform":

                    poddd = POD('svd', rank = dim)
                    rom = ROM(db_training, poddd, ann_POD)

                # db training lo facciamo sullo stesso set ma la base ridotta la troviamo con l'encoder della conv AE
                    start_rom = time()
                    rom.fit()
                    end_rom = time()
                    print("time to train rom = ", end_rom - start_rom)

                    
                    
                    
                    # r_rom_train = rom.reduction.transform(db_training.parameters.T)
                    # e_rom_train = rom.reduction.inverse_transform(r_rom_train.T).T
                    # e_rom_train = decoded_rom_train.squeeze().reshape(180,65536)
                    # r_rom_test = rom.predict(params_testing)
                    # e_rom_test = rom.reduction.inverse_transform(r_rom_test).T

                    
                    for ii in range(len(train_FOM)):
                        pred_sol = rom.predict(params_training[ii])
                        train_err_POD_NN[ii] = np.linalg.norm(train_POD[ii] - pred_sol)/np.linalg.norm(train_POD[ii])

                    for jj in range(len(test_FOM)):
                        pred_sol = rom.predict(params_testing[jj])
                        test_err_POD_NN[jj] = np.linalg.norm(test_POD[jj] - pred_sol)/np.linalg.norm(test_POD[jj])

                    
                    idx_train = params_training[-1]
                    idx_test = params_testing[-1]
                    real_rom_train = rom.predict(idx_train).reshape((256, 256))
                    real_rom_test = rom.predict(idx_test).reshape((256, 256))
                    plt.figure()

                    plt.imshow(real_rom_train,cmap=plt.cm.jet,origin='lower')
                    plt.title(f'Reconstruction{type_err}')
                    plt.colorbar()
                    plt.clim(0.9, 1.7)
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_train_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    plt.figure()

                    plt.imshow(real_rom_test,cmap=plt.cm.jet,origin='lower')
                    plt.title(f'Reconstruction{type_err}')
                    plt.colorbar()
                    plt.clim(0.9, 1.7)
                    plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_test_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()
                    #plot error TRAINING
                    # plt.figure(figsize=(15, 15))
                    # plt.subplot(2,2,1)
                    # plt.semilogy(params_training, train_err_POD_NN, 'bx-')
                    # plt.title(f"{type_err}_error_Train")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time")

                    # plt.subplot(2,2,2)
                    # plt.semilogy(params_training, train_err_conv, 'bo-')
                    # plt.title("convAE_error_Train")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time") 

                    # #plot error TESTING
                    # plt.subplot(2,2,3)
                    # plt.semilogy(params_testing, test_err_POD_NN, 'rx-')
                    # plt.title(f"{type_err}_error_Test")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time")

                    # plt.subplot(2,2,4)
                    # plt.semilogy(params_testing, test_err_conv, 'ro-')
                    # plt.title("convAE_error_Test")
                    # plt.ylabel("relative error")
                    # plt.xlabel("time")
                    
                    
                    # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_four.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    # plt.close()


                    # plt.figure()
                    # plt.semilogy(params_training, train_err_POD_NN, 'bx-', )
                    # plt.semilogy(params_testing, test_err_POD_NN, 'ro-')
                    # plt.legend(['Train', 'Test'])
                    # #plt.title(f"{type_err}_error_Train")
                    # #plt.ylabel("relative error")
                    # #plt.xlabel("time")

                    # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    # plt.close()

            if type_err == "NN_encoder-FOM":
                
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                latent_sol = conv_ae.transform(train_torch)
                          
                ann_enc.fit(params_training,latent_sol.T)

                # db training lo facciamo sullo stesso set ma la base ridotta la troviamo con l'encoder della conv AE
                
                pred_sol_train = [] #np.zeros_like((train_POD))
                pred_sol_test = []

                # for ii in range(len(train_FOM)):

                #     pred_sol_NN_enc_train = ann.predict(params_training[ii]) 
                #     #print("pred_sol_NN_enc_train shape", pred_sol_NN_enc_train.shape)
                #     pred_sol_train.append(np.array(pred_sol_NN_enc_train))
                
                pred_sol_train = ann_enc.predict(params_training) 

                reconstruct_NN_train = conv_ae.inverse_transform(np.array(pred_sol_train).T).T.squeeze()
                print("shape of reconstruct_NN", reconstruct_NN_train.shape)
                

                pred_sol_test = ann_enc.predict(params_testing) 
                
                reconstruct_NN_test = conv_ae.inverse_transform(np.array(pred_sol_test).T).T.squeeze()
                reconstruct_NN_test.reshape(20, 256*256)
                #print("shape of reconstruct NN", pred_sol_train.shape)

                for ii in range(len(train_FOM)):
                    train_err_NN_encoder[ii] = np.linalg.norm(train_POD[ii] - reconstruct_NN_train.reshape(180, 256*256)[ii])/np.linalg.norm(train_POD[ii]) 

                for jj in range(len(test_FOM)):
                    test_err_NN_encoder[jj] = np.linalg.norm(test_POD[jj] - reconstruct_NN_test.reshape(20, 256*256)[jj])/np.linalg.norm(test_POD[jj]) 
                
           
                real_NNenc_train = reconstruct_NN_train[-1].reshape((256, 256))
                real_NNenc_test = reconstruct_NN_test[-1].reshape((256, 256))

                plt.figure()

                plt.imshow(real_NNenc_train,cmap=plt.cm.jet,origin='lower')
                plt.title(f'Reconstruction{type_err}')
                plt.colorbar()
                plt.clim(0.9, 1.7)
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_train_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()
                plt.figure()

                plt.imshow(real_NNenc_test,cmap=plt.cm.jet,origin='lower')
                plt.title(f'Reconstruction{type_err}')
                plt.colorbar()
                plt.clim(0.9, 1.7)
                plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Reconstruction_{type_err}_test_last.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                plt.close()
                
                #plot error TRAINING
                # plt.figure(figsize=(15, 15))
                # plt.subplot(1,2,1)
                # plt.semilogy(params_training, train_err_NN_encoder, 'bx-')
                # plt.title(f"{type_err}_error_Train")
                # plt.ylabel("relative error")
                # plt.xlabel("time")

                # plt.subplot(1,2,2)
                # plt.semilogy(params_training, train_err_conv, 'bo-')
                # plt.title("convAE_error_Train")
                # plt.ylabel("relative error")
                # plt.xlabel("time") 

                # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_vs_convAE_trainPROVA.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                # plt.close()

                # #plot error TESTING
                # plt.figure(figsize=(15, 15))
                # plt.subplot(1,2,1)
                # plt.semilogy(params_testing, test_err_NN_encoder, 'rx-')
                # plt.title(f"{type_err}_error_Test")
                # plt.ylabel("relative error")
                # plt.xlabel("time")

                # plt.subplot(1,2,2)
                # plt.semilogy(params_testing, test_err_conv, 'ro-')
                # plt.title("convAE_error_Test")
                # plt.ylabel("relative error")
                # plt.xlabel("time")
                
                
                # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_vs_convAE_test.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                # plt.close()


                # plt.figure()
                # plt.semilogy(params_training, train_err_NN_encoder, 'bx-', )
                # plt.semilogy(params_testing, test_err_NN_encoder, 'ro-')
                # plt.legend(['Train', 'Test'])
                # #plt.title(f"{type_err}_error_Train")
                # #plt.ylabel("relative error")
                # #plt.xlabel("time")

                # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                # plt.close()


        plt.figure(figsize=(15, 15))
        
        plt.semilogy(params_testing, test_err_conv, "s")
        plt.semilogy(params_testing, test_err_POD_NN, "o")
        plt.semilogy(params_testing, test_err_NN_encoder, "d")
        plt.semilogy(params_training, train_err_POD, ".") 
        plt.semilogy(params_training, train_err_POD_NN, "1")
        plt.semilogy(params_training, train_err_conv, "|")
        plt.semilogy(params_training, train_err_NN_encoder, "x")
        
        plt.legend(["test_convAE", "test_POD_NN","test_NN+encoder","train_POD_projection", "train_POD_NN", "train_convAE","train_NN+encoder",])
        #plt.title(f"{type_err}_error_Train")
        plt.ylabel("relative error")
        plt.xlabel("time")
        plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Errors.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
        plt.close()