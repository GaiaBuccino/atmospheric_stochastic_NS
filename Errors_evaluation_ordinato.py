### Evaluating errors
"""
Module for FNN-Autoencoders

"""
import pandas as pd
import conv_ae
import pod
from conv_ae import convAE
import torch
from torch import nn
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import Database, ANN, AE
from ezyrb import ReducedOrderModel as ROM
import pickle
import copy
import os
from time import time
from typing import Optional, Tuple

#Different test possibilities: discontinuity, snapshots

def prepare_data(db_type: str, folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """load of the data to be processed

    Args:
        db_type (str): name of the case to be loaded
        folder (str): name of the path containing the data

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: loaded data organized as train_dataset, test_dataset, train_parameters, test_parameters
    """
    print(f"Dealing with {db_type}...")
    train_FOM = np.load(os.path.join(folder, f'{db_type}_training.npy'))     
    test_FOM = np.load(os.path.join(folder, f'{db_type}_testing.npy'))   
    params_training = np.load(os.path.join(folder, f'{db_type}_params_training.npy'))
    params_testing = np.load(os.path.join(folder, f'{db_type}_params_testing.npy'))

    return train_FOM, test_FOM, params_training, params_testing

def perform_convAE(train_dataset: np.ndarray, test_dataset: np.ndarray, rank:int, dump_path: str, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """_summary_

    Args:
        train_FOM (np.ndarray): train dataset with the shape (number of samples, number of channels, first dimension, second dimension)
        test_FOM (np.ndarray): train dataset with the shape (number of samples, number of channels, first dimension, second dimension)
        dump_path (str): path where the convAe trained model is saved

    Returns:
        error.squeeze(np.ndarray): vector containing the errors between the FOM simulation and the convAE approximation
    """

    tensor_test = np.expand_dims(test_dataset, axis=1) 
    tensor_train = np.expand_dims(train_dataset, axis=1)     

    if os.path.exists(dump_path):
        
        conv_ae = torch.load(dump_path)   
        #trovare un modo per salvare il tempo di training

    else:
        
        fake_f = torch.nn.ELU
        #optim = torch.optim.Adam
        conv_layers = 6
        epochs = 80
        fake_val = 2
        neurons_linear = fake_val

        # Pod_type = 'classical'
        # if weights is not None:
        #     Pod_type = 'weighted'

        conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, neurons_linear)

        train_time = -time.perf_counter()
        conv_ae.fit(tensor_train, weights)
        train_time += time.perf_counter()

        torch.save(conv_ae, dump_path)

    #do testing
    pred_train = conv_ae.inverse_transform(conv_ae.transform(tensor_train))
    error_train = pred_train - tensor_train

    pred_test = conv_ae.inverse_transform(conv_ae.transform(tensor_test))
    error_test = pred_test - tensor_test


    return error_train.squeeze(), error_test.squeeze(), train_time


def perform_POD(train_dataset: np.ndarray, test_dataset: np.ndarray, rank: int, dump_path: str, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """_summary_

    Args:
        snapshots (np.ndarray): snapshots we wanto to perform the POD on
        rank (int): reduced basis cardinality
        weights (np.ndarray): optional weigths for wPOD, if not given performs classical POD
    Returns:        
        Pod_type (str): value indicating the type of POD performed (weighted, classical)
        errors (np.ndarray): vector containing the errors between the FOM simulation and the POD approximation
    """
    #reshape keeps the number of elements constant
    #since the number  of elements in the first dimension remains the same, the remaining dim has n_elements/n_rows 
    #(n_elements = total elements in the structure)
    method = 'svd'
    # Pod_type = 'classical'
    if weights is not None:
        method = 'correlation_matrix'
        # Pod_type = 'weighted'
    
    train_POD = train_dataset.reshape(len(train_dataset), -1)
    test_POD = test_dataset.reshape(len(test_dataset), -1)
    Pod = pod.POD(method, weights = weights, rank = rank)    
    train_time = -time.perf_counter()
    Pod.fit(train_POD.T)
    train_time += time.perf_counter()

    pred_train = Pod.inverse_transform(Pod.transform(train_POD.T)).T
    error_train = pred_train - train_dataset

    pred_test = Pod.inverse_transform(Pod.transform(test_POD.T)).T
    error_test = pred_test - test_dataset

    return error_train, error_test, train_time


def perform_POD_NN(train_dataset: np.ndarray, test_dataset: np.ndarray, params_training:np.ndarray, params_testing:np.ndarray, rank:int, ann:ANN, dump_path:str, weights: Optional[np.ndarray] = None)  -> Tuple[np.ndarray, np.ndarray, float]:
    """
    perform the POD method learning the coefficients with a neural network

    Args:
        train_FOM (np.ndarray): train dataset
        test_FOM (np.ndarray): test dataset
        params_training (np.ndarray): train parameters
        params_testing (np.ndarray): test parameters
        method (str): method to perform the POD
        rank (int): cardinality of the reduced basis
        ann (ezyrb.ann.ANN): structure of the neural network

    Returns:
        Tuple[np.ndarray, np.ndarray]: [error_train, error_test] 
    """
    train_data = train_dataset.reshape(len(train_dataset), -1)
    test_data = test_dataset.reshape(len(test_dataset), -1)
    if os.path.exists(dump_path):
        
        rom = ROM.load(dump_path)   
        #trovare il modo di salvare il tempo di training

    else:
        db_train = Database(params_training, train_data)
        
        method = 'svd'
        # Pod_type = 'classical'
        if weights is not None:
            method = 'correlation_matrix'
            # Pod_type = 'weighted'
        rpod = pod.POD(method, weights=weights rank = rank)
        rom = ROM(db_train, rpod, ann)

        # db training lo facciamo sullo stesso set ma la base ridotta la troviamo con l'encoder della conv AE
        train_time = -time.perf_counter()
        rom.fit()
        train_time += time.perf_counter()
        
        rom.save(dump_path, save_db = False)
    #print("time to train rom = ", end_rom - start_rom)

    #compute errors
    pred_train = rom.predict(params_training)
    error_train = pred_train - train_data
    pred_test = rom.predict(params_testing)
    error_test = pred_test - test_data

    return error_train, error_test, train_time




types_test = ["synthetic_discontinuity", "simulated_gulf"]  #"snapshots" or "discontinuity"

for test in types_test:

    ### DATA LOADING ###

    # if test == "snapshots":
        
    #     print("DEALING WITH SNAPSHOTS...")

    #     train_FOM = np.load('Data/snap_w_training.npy')     #(180, 256, 256)
    #     test_FOM = np.load('Data/snap_w_testing.npy')       #(20, 256, 256)
    #     params_training = np.load('Data/params_training.npy')   #(180, 1)
    #     params_testing = np.load('Data/params_testing.npy')     #(20, 1)
        
    # elif test == "discontinuity":

    #     print("DEALING WITH DISCONTINUITY...")

    #     train_FOM = np.load('Data/discontinuity_training.npy')      #(180, 256, 256)
    #     test_FOM = np.load('Data/discontinuity_testing.npy')
    #     params_training = np.load('Data/disc_params_training.npy')
    #     params_testing = np.load('Data/disc_params_testing.npy')

    train_dataset, test_dataset, params_training, params_testing = prepare_data(test, 'Data')

    #weights = np.ones(len(params_training))
    #weights = np.load("weights.npy")
    weights = beta.pdf((params_training - 0.3) / 2.7, 5, 2).squeeze() / 2.7

    path = f'./Stochastic_results/{test}_tests'
  
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(error) 

    ranks = [14]    # 2, 6, 14, 30
    models = ["POD", "wPOD", "POD-NN", "wPOD-NN", "convAE","wconvAE"]
    
    for model in models:

        for rank in ranks:

            directory = f'/{model}_model/rank_{rank}.pt' 
            path = os.path.join(path, directory)
            
                
            if model == "POD":

                train_err, test_err, time = perform_POD(train_dataset, test_dataset, rank, path)
             
                POD_statistics = {'training error': train_err,
                                  'testing error': test_err,
                                  'training time': time} 
                
            elif model == "wPOD":

                train_err, test_err, time = perform_POD(train_dataset, test_dataset, rank, path, weights = weights)
             
                wPOD_statistics = {'training error': train_err,
                                  'testing error': test_err,
                                  'training time': time}


                

            ann_POD = ANN([16,64,64], nn.Tanh(), [60000, 1e-12])     #da correggere in 60000
            ann_enc = ANN([16,64,64], nn.Tanh(), [60000, 1e-12])

            #initialization UQ tests
            eee_train = e_train.reshape(180, 65536)
            UQ_test = 180
            sol_conv = np.zeros((UQ_test,256*256))
            sol_POD = np.zeros((UQ_test,256*256))
            sol_FOM = np.zeros((UQ_test,256*256))
            exp_FOM = np.zeros(256*256)
            exp_POD = np.zeros(256*256)
            exp_convAE = np.zeros(256*256)
            exp_wconvAE = np.zeros(256*256)
            exp_wPOD = np.zeros(256*256)
            var_FOM = np.zeros(256*256)
            var_POD = np.zeros(256*256)
            var_convAE = np.zeros(256*256)
            var_wconvAE = np.zeros(256*256)
            var_wPOD = np.zeros(256*256)

            for kk in range(UQ_test):

                sol_FOM[:][kk] = train_POD[kk][:]*weights[kk]
            
            exp_FOM= np.sum(sol_FOM, axis=0)/np.sum(weights)
            

            simul_folder = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{rank}linear_neurons/'
            UQ_folder = simul_folder+'UQ/'
            os.system(f"mkdir {UQ_folder}")

            plt.imshow(exp_FOM.reshape(256,256),cmap=plt.cm.jet,origin='lower')

            plt.colorbar()
            plt.savefig(UQ_folder+'exp_FOM', format='pdf',bbox_inches='tight',pad_inches = 0)
            plt.close()

            
            
            
            for type_err in models:

                if type_err == "convAE-FOM":

                    for kk in range(UQ_test):

                        sol_conv[:][kk] = eee_train[kk][:]*weights[kk]

                    for ii in range(len(train_dataset)):
                        
                        train_err_conv[ii] = np.linalg.norm(e_train[ii] - train_dataset[ii])/np.linalg.norm(train_dataset[ii])  #conv_AE reconstructed VS FOM
                        
                    for jj in range(len(test_dataset)):
                    
                        test_err_conv[jj] = np.linalg.norm(e_test[jj] - test_dataset[jj])/np.linalg.norm(test_dataset[jj])

                    #for dof in range(eee_train.shape[1]):
                        
                    exp_convAE= np.sum(sol_conv, axis=0)/np.sum(weights) 

                    simul_folder = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{rank}linear_neurons/'
                    UQ_folder = simul_folder+'UQ/'
                    os.system(f"mkdir {UQ_folder}")

                    plt.imshow(exp_convAE.reshape(256,256),cmap=plt.cm.jet,origin='lower')

                    plt.colorbar()
                    plt.savefig(UQ_folder+'exp_convAE', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()   


                # if type_err == "POD-inv_transform" or type_err == "POD_NN-inv_transform":
                    
                #     # POD model
                #     print("in POD-inv or POD-NN type_err = ", type_err)
                #     podd = pod.POD('svd', rank = dim)    #con 'correlation matrix' viene identica
                #     podd.fit(db_training.snapshots.T)
                #     pod_r_train = podd.transform(db_training.snapshots.T).T
                #     print("shape rom_r_train", pod_r_train.shape)
                #     pod_e_train = podd.inverse_transform(pod_r_train.T).T
                    
                #     simul_folder = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/'
                #     modes_folder = simul_folder+'pod/'

                #     os.system(f"mkdir {modes_folder}")
                #     for i in range(podd.modes.shape[1]):
                #         plt.imshow(podd.modes[:,i].reshape(256,256),cmap=plt.cm.jet,origin='lower')

                #         plt.colorbar()
                #         plt.savefig(modes_folder+'mode_%02d.pdf'%i, format='pdf',bbox_inches='tight',pad_inches = 0)
                #         plt.close()   


                #     plt.imshow(db_training.snapshots[0].reshape(256,256),cmap=plt.cm.jet,origin='lower')
                #     plt.colorbar()
                #     plt.savefig(simul_folder+'snap_pod_FOM_0.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                #     plt.close()   
                #     plt.imshow(pod_e_train[0].reshape(256,256),cmap=plt.cm.jet,origin='lower')
                #     plt.colorbar()
                #     plt.savefig(simul_folder+'snap_pod_rec_0.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                #     plt.close()  


                #     #test_err_POD = np.zeros(len(test_POD))

                #     if type_err == "POD-inv_transform":

                #         for ii in range(len(train_FOM)):
                        
                #             train_err_POD[ii] = np.linalg.norm(train_POD[ii] - pod_e_train[ii])/np.linalg.norm(train_POD[ii])     # POD reconstructed VS FOM
                            
                #         for kk in range(UQ_test):

                #             sol_POD[:][kk] = pod_e_train[kk][:]*weights[kk]

                #         exp_POD= np.sum(sol_POD, axis=0)/np.sum(weights) 

                #         simul_folder = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/'
                #         UQ_folder = simul_folder+'UQ/'
                #         os.system(f"mkdir {UQ_folder}")

                #         plt.imshow(exp_POD.reshape(256,256),cmap=plt.cm.jet,origin='lower')

                #         plt.colorbar()
                #         plt.savefig(UQ_folder+'exp_POD', format='pdf',bbox_inches='tight',pad_inches = 0)
                #         plt.close()   
                    


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

                        
                    # elif type_err == "POD_NN-inv_transform":

                    #     rom_pod = pod.POD('svd', rank = dim)
                    #     rom = ROM(db_training, rom_pod, ann_POD)

                    # # db training lo facciamo sullo stesso set ma la base ridotta la troviamo con l'encoder della conv AE
                    #     start_rom = time()
                    #     rom.fit()
                    #     end_rom = time()
                    #     print("time to train rom = ", end_rom - start_rom)

                    #     # r_rom_train = rom.reduction.transform(db_training.parameters.T)
                    #     # e_rom_train = rom.reduction.inverse_transform(r_rom_train.T).T
                    #     # e_rom_train = decoded_rom_train.squeeze().reshape(180,65536)
                    #     # r_rom_test = rom.predict(params_testing)
                    #     # e_rom_test = rom.reduction.inverse_transform(r_rom_test).T
                        
                    #     for ii in range(len(train_FOM)):
                    #         pred_sol = rom.predict(params_training[ii])
                    #         train_err_POD_NN[ii] = np.linalg.norm(train_POD[ii] - pred_sol)/np.linalg.norm(train_POD[ii])

                    #     for jj in range(len(test_FOM)):
                    #         start = time()
                    #         pred_sol = rom.predict(params_testing[jj])  #passare tutto il set di dati insieme -> start time, predict e stop da posizionare prima del for loop
                    #         stop = time()-start #salvare in un excel
                    #         test_err_POD_NN[jj] = np.linalg.norm(test_POD[jj] - pred_sol)/np.linalg.norm(test_POD[jj])

                    #     #plot error TRAINING
                    #     # plt.figure(figsize=(15, 15))
                    #     # plt.subplot(2,2,1)
                    #     # plt.semilogy(params_training, train_err_POD_NN, 'bx-')
                    #     # plt.title(f"{type_err}_error_Train")
                    #     # plt.ylabel("relative error")
                    #     # plt.xlabel("time")

                    #     # plt.subplot(2,2,2)
                    #     # plt.semilogy(params_training, train_err_conv, 'bo-')
                    #     # plt.title("convAE_error_Train")
                    #     # plt.ylabel("relative error")
                    #     # plt.xlabel("time") 

                    #     # #plot error TESTING
                    #     # plt.subplot(2,2,3)
                    #     # plt.semilogy(params_testing, test_err_POD_NN, 'rx-')
                    #     # plt.title(f"{type_err}_error_Test")
                    #     # plt.ylabel("relative error")
                    #     # plt.xlabel("time")

                    #     # plt.subplot(2,2,4)
                    #     # plt.semilogy(params_testing, test_err_conv, 'ro-')
                    #     # plt.title("convAE_error_Test")
                    #     # plt.ylabel("relative error")
                    #     # plt.xlabel("time")
                        
                        
                    #     # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}_four.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    #     # plt.close()


                    #     # plt.figure()
                    #     # plt.semilogy(params_training, train_err_POD_NN, 'bx-', )
                    #     # plt.semilogy(params_testing, test_err_POD_NN, 'ro-')
                    #     # plt.legend(['Train', 'Test'])
                    #     # #plt.title(f"{type_err}_error_Train")
                    #     # #plt.ylabel("relative error")
                    #     # #plt.xlabel("time")

                    #     # plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{dim}linear_neurons/Error_{type_err}.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    #     # plt.close()

                if type_err == "weigthedPOD-inv_transform":
                    #weights = np.ones(len(params_training)) 
                    #weights = np.load("weights.npy")
                    weights = beta.pdf((db_training.parameters-0.3)/2.7,5,2).squeeze()/2.7

                    print("weights shape", weights.shape)             
                    
                    # wPOD model
                    wpod = pod.POD('correlation_matrix', rank = rank)     
                    print("weights = ", weights)
                    wpod.fit(db_training.snapshots.T, weights)

                    print("modes = ",wpod.modes.shape)  # (65536,14)
                    
                    wpod_r_train = wpod.transform(db_training.snapshots.T).T
                    wpod_e_train = wpod.inverse_transform(wpod_r_train.T).T

                    simul_folder = f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{rank}linear_neurons/'
                    modes_folder = simul_folder+'wpod/'
                    os.system(f"mkdir {modes_folder}")

                    plt.plot(db_training.parameters, weights, '.')
                    plt.savefig(simul_folder+'param_weights.pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    for i in range(wpod.modes.shape[1]):
                        plt.imshow(wpod.modes[:,i].reshape(256,256),cmap=plt.cm.jet,origin='lower')

                        plt.colorbar()
                        plt.savefig(modes_folder+'mode_%02d.pdf'%i, format='pdf',bbox_inches='tight',pad_inches = 0)
                        plt.close()   

                    plt.imshow(db_training.snapshots[0].reshape(256,256),cmap=plt.cm.jet,origin='lower')
                    plt.colorbar()
                    plt.savefig(simul_folder+'snap_wpod_FOM_0.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()   
                    plt.imshow(wpod_e_train[0].reshape(256,256),cmap=plt.cm.jet,origin='lower')
                    plt.colorbar()
                    plt.savefig(simul_folder+'snap_wpod_rec_0.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
                    plt.close()   

                    for ii in range(len(train_dataset)):
                    #controllare se bisogna moltiplicare o meno l'errore in se' per il peso    
                        train_err_wPOD[ii] = np.linalg.norm(train_POD[ii] - wpod_e_train[ii])/np.linalg.norm(train_POD[ii])
                        
                    weighted_error = np.sqrt(np.sum(weights @ train_err_wPOD**2))/np.sqrt(np.sum(weights))  

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
                    

                    for jj in range(len(test_dataset)):

                        pred_sol_NN_enc_test = ann_enc.predict(params_testing[jj]) 
                        #print("pred_sol_NN_enc_test shape", pred_sol_NN_enc_test.shape)
                        pred_sol_test.append(np.array(pred_sol_NN_enc_test))
                    
                    reconstruct_NN_test = conv_ae.inverse_transform(np.array(pred_sol_test).T).T.squeeze()
                    reconstruct_NN_test.reshape(20, 256*256)
                    #print("shape of reconstruct NN", pred_sol_train.shape)

                    for ii in range(len(train_dataset)):
                        train_err_NN_encoder[ii] = np.linalg.norm(train_POD[ii] - reconstruct_NN_train.reshape(180, 256*256)[ii])/np.linalg.norm(train_POD[ii]) 

                    for jj in range(len(test_dataset)):
                        test_err_NN_encoder[jj] = np.linalg.norm(test_POD[jj] - reconstruct_NN_test.reshape(20, 256*256)[jj])/np.linalg.norm(test_POD[jj]) 

            
            plt.figure(figsize=(15, 15))
            
            plt.semilogy(params_testing, test_err_conv, "s")
            plt.semilogy(params_testing, test_err_POD_NN, "o")
            plt.semilogy(params_testing, test_err_NN_encoder, "d")
            plt.semilogy(params_training, train_err_POD, "*") 
            plt.semilogy(params_training, train_err_wPOD, ".") 
            plt.semilogy(params_training, train_err_POD_NN, "1")
            plt.semilogy(params_training, train_err_conv, "|")
            plt.semilogy(params_training, train_err_NN_encoder, "x")
            
            plt.legend(["test_convAE", "test_POD_NN","test_NN+encoder","train_POD_projection", "train_wPOD_projection", "train_POD_NN", "train_convAE","train_NN+encoder"])
            #plt.title(f"{type_err}_error_Train")
            plt.ylabel("relative error")
            plt.xlabel("time")
            plt.savefig(f'./Stochastic_results/{test}_tests/{test}_{conv_layers}_conv/conv_AE_{epochs}epochs/{rank}linear_neurons/Errors.pdf', format='pdf',bbox_inches='tight',pad_inches = 0)
            plt.close()

            #print("shape wpod_modes= ", wpod.modes.shape) #65k,14
            """  aa = np.arange(len(wpod.modes[:,0]))

            print("mode size",wpod.modes[:,0].shape)
            """

        
            