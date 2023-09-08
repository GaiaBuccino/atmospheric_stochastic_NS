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
from time import perf_counter
from typing import Optional, Tuple
import csv

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

def perform_convAE(train_dataset: np.ndarray, test_dataset: np.ndarray, rank:int, dump_path: str, weights: Optional[np.ndarray] = None) -> dict:
    """_summary_

    Args:
        train_FOM (np.ndarray): train dataset with the shape (number of samples, number of channels, first dimension, second dimension)
        test_FOM (np.ndarray): train dataset with the shape (number of samples, number of channels, first dimension, second dimension)
        dump_path (str): path where the convAe trained model is saved

    Returns:
        dict: TBD
    """

    tensor_train = np.expand_dims(train_dataset, axis=1)     
    tensor_test = np.expand_dims(test_dataset, axis=1) 
    
    if os.path.exists(dump_path):
        
        conv_ae = torch.load(dump_path)   
        #trovare un modo per salvare il tempo di training

    else:
        
        fake_f = torch.nn.ELU
        #optim = torch.optim.Adam
        conv_layers = 6
        epochs = 30
        fake_val = 2
        neurons_linear = fake_val

        conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, neurons_linear)

        train_time = -perf_counter()
        conv_ae.fit(tensor_train)
        train_time += perf_counter()

        #torch.save(conv_ae, dump_path)

    #do testing
    pred_train = conv_ae.inverse_transform(conv_ae.transform(tensor_train)).T
    error_train = pred_train - tensor_train

    pred_test = conv_ae.inverse_transform(conv_ae.transform(tensor_test)).T
    error_test = pred_test - tensor_test

    convAE_type = 'classical'
    E_value = []
    variance = 0

    if weights is not None:
        convAE_type = 'weighted'
        E_value = compute_expected_value(pred_train.reshape(len(train_dataset),-1), weights)
        variance = compute_variance(train_dataset.reshape(len(train_dataset),-1), pred_train.reshape(len(train_dataset), -1), weights)

    statistics = {'model' : f'{convAE_type} convAE',
                  'rank' : rank,
                  'training error': error_train,
                  'testing error': error_test,
                  'training time': train_time,
                  'expected value' : E_value,
                  'variance': variance} 
    
    print(f"{statistics['model']} performed...")

    return statistics

def perform_POD(train_dataset: np.ndarray, test_dataset: np.ndarray, rank: int, weights: Optional[np.ndarray] = None) -> dict:
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
    Pod_type = 'classical'
    
    if weights is not None:
        method = 'correlation_matrix'
        Pod_type = 'weighted'
    
    train_POD = train_dataset.reshape(len(train_dataset), -1)
    test_POD = test_dataset.reshape(len(test_dataset), -1)
    Pod = pod.POD(method, weights = weights, rank = rank)    
    train_time = -perf_counter()
    Pod.fit(train_POD.T)
    train_time += perf_counter()

    pred_train = Pod.inverse_transform(Pod.transform(train_POD.T)).T
    error_train = pred_train - train_POD

    pred_test = Pod.inverse_transform(Pod.transform(test_POD.T)).T
    error_test = pred_test - test_POD

    E_value = []
    variance = 0

    if Pod_type == 'weighted':
        E_value = compute_expected_value(pred_train, weights)
        variance = compute_variance(train_POD, pred_train, weights)

    statistics = {'rank' : rank,
                  'training error': error_train,
                  'testing error': error_test,
                  'training time': train_time,
                  'expected value': E_value,
                  'variance': variance,
                  'model_path': None} 

    print(f"{Pod_type} POD performed...")

    return statistics

def perform_POD_NN(train_dataset: np.ndarray, test_dataset: np.ndarray, params_training:np.ndarray, params_testing:np.ndarray, trained: dict, rank:int, ann:ANN, dump_path:str, model_path: str, weights: Optional[np.ndarray] = None)  -> dict:
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

    method = 'svd'
    Pod_type = 'classical'

    if weights is not None:
        method = 'correlation_matrix'
        Pod_type = 'weighted'

    # if os.path.exists(dump_path):
        
    #     rom = ROM.load(dump_path)
    #     train_time = trained[model_path] 
    #     #trovare il modo di salvare il tempo di training

    # else:
    db_train = Database(params_training, train_data)
    rpod = pod.POD(method, weights=weights, rank = rank)
    rom = ROM(db_train, rpod, ann)

    train_time = -perf_counter()
    rom.fit()
    train_time += perf_counter()

    rom.save(dump_path, save_db = False)
    trained[model_path] = train_time

    #compute errors
    pred_train = rom.predict(params_training)
    error_train = pred_train - train_data
    pred_test = rom.predict(params_testing)
    error_test = pred_test - test_data

    E_value = []
    variance = 0

    if Pod_type == 'weighted':
        E_value = compute_expected_value(pred_train.reshape(len(pred_train), -1), weights)
        variance = compute_variance(train_dataset.reshape(len(pred_train), -1), pred_train, weights)

    statistics = {'rank' : rank,
                  'training error': error_train,
                  'testing error': error_test,
                  'training time': train_time,
                  'expected value': E_value,
                  'variance': variance,
                  'model_path': dump_path} 

    print(f"{Pod_type} POD-NN performed...")
    
    return statistics

def perform_NN_encoder(train_dataset: np.ndarray, test_dataset: np.ndarray, params_training:np.ndarray, params_testing:np.ndarray, rank:int, ann:ANN, dump_path:str, weights: Optional[np.ndarray] = None)  -> dict:    

    tensor_test = np.expand_dims(test_dataset, axis=1) 
    tensor_train = np.expand_dims(train_dataset, axis=1)     

    # if os.path.exists(dump_path):
        
    #     print("Loading existing convAE")
    #     conv_ae = torch.load(dump_path)   
    #     #trovare un modo per salvare il tempo di training

    # else:
          
    fake_f = torch.nn.ELU
    #optim = torch.optim.Adam
    conv_layers = 6
    epochs = 30
    fake_val = 2
    neurons_linear = fake_val

    conv_ae = convAE([fake_val], [fake_val], fake_f(), fake_f(), epochs, neurons_linear)

    train_time = -perf_counter()
    conv_ae.fit(tensor_train)
    train_time += perf_counter()
    torch.save(conv_ae, dump_path)
    
    reduced_train = conv_ae.transform(tensor_train)
    ann_enc.fit(params_training,reduced_train.T)

    pred_train = conv_ae.inverse_transform(np.array(ann_enc.predict(params_training)).T).T.squeeze()
    error_train = pred_train - tensor_train

    pred_test = conv_ae.inverse_transform(np.array(ann_enc.predict(params_testing)).T).T.squeeze()
    error_test = pred_test - tensor_test

    convAE_type = 'classical'
    E_value = []
    variance = 0

    if weights is not None:
        convAE_type = 'weighted'
        E_value = compute_expected_value(pred_train.reshape(len(train_dataset), -1), weights)
        variance = compute_variance(train_dataset.reshape(len(train_dataset), -1), pred_train.reshape(len(train_dataset), -1), weights)

    statistics = {'rank' : rank,
                  'training error': error_train,
                  'testing error': error_test,
                  'training time': train_time,
                  'expected value': E_value,
                  'variance': variance,
                  'model_path': dump_path} 

    print(f"{convAE_type} convAE performed...")

    return statistics

def compute_expected_value(dataset: np.ndarray, weights: np.ndarray) -> np.ndarray :

    weighted_dataset = np.zeros((dataset.shape[0],dataset.shape[1]))

    for kk in range(dataset.shape[0]):
        weighted_dataset[kk][:] = dataset[kk][:]*weights[kk]
    
    E_value = np.sum(weighted_dataset, axis=0)/np.sum(weights)

    return E_value
    
def compute_variance(hf_dataset: np.ndarray, lf_dataset:np.ndarray, weigths: np.ndarray) -> np.ndarray:

    weighted_dataset = np.zeros(hf_dataset.shape[1])

    for kk in range(hf_dataset.shape[0]):
        weighted_dataset[:][kk] = np.linalg.norm((hf_dataset[kk][:] - lf_dataset[kk][:])*np.sqrt(weights[kk]))
    
    var = np.sum(weighted_dataset, axis=0)/np.sum(weights)

    return var



### main ###

types_test = ["synthetic_discontinuity"]#, "simulated_gulf"]  #"snapshots" or "discontinuity"

for test in types_test:

    train_dataset, test_dataset, params_training, params_testing = prepare_data(test, 'Data')

    #weights = np.ones(len(params_training))
    #weights = np.load("weights.npy")
    weights = beta.pdf((params_training - 0.3) / 2.7, 5, 2).squeeze() / 2.7

    path = f"./Stochastic_results/{test}_tests/"
  
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(error) 

    ranks = [14]    # 2, 6, 14, 30

    # if os.path.exists(f'{path}/Trained_models.csv'):
    #trained_models = {}#pd.read_csv('trained_models.csv')
    models = ["POD", "wPOD", "POD_NN", "wPOD_NN", "convAE", "wconvAE", "NN_encoder", "NN_wencoder"]
    statistics = []
    
    for model in models:

        for rank in ranks:

            directory = f"{model}_model/rank_{rank}" 
            folders_path = os.path.join(path, directory)
            try:
                os.makedirs(folders_path, exist_ok=True)
            except OSError as error:
                print(error) 
            
            #if os.path.exists(f'{path}/Trained_models.csv'):

            if os.path.exists(f'{path}/Trained_models.csv'):
                trained_models = pd.read_csv(f'{path}/Trained_models.csv')
            else:
                trained_models = pd.DataFrame(columns=['model','rank', 'training error', 
                                                       'testing error', 'training time', 'expected value', 
                                                       'variance', 'model_path'])
            
            print(f"performing {model}...")
            df_model = trained_models.query(f"model == '{model}' and rank == {rank}")    
            
            if len(df_model) == 0:

                print(f"performing {model}...")

                if model == "POD":

                    model_stats = perform_POD(train_dataset, test_dataset, rank)
                    
                elif model == "wPOD":

                    model_stats = perform_POD(train_dataset, test_dataset, rank, weights)

                elif model == "POD_NN":

                    print(f"performing {model}...")
                    epochs = 110
                    ann_POD = ANN([16,64,64], nn.Tanh(), [epochs, 1e-12])
                    model_path = f"{model}_{epochs}.rom"
                    new_path = os.path.join(folders_path, model_path)
                    model_stats = perform_POD_NN(train_dataset, test_dataset, params_training, params_testing, rank, ann_POD, new_path)
                    
                elif model == "wPOD_NN":

                    print(f"performing {model}...")
                    epochs = 110
                    ann_POD = ANN([16,64,64], nn.Tanh(), [epochs, 1e-12])
                    model_path = f"{model}_{epochs}.rom"
                    new_path = os.path.join(folders_path, model_path)
                    model_stats = perform_POD_NN(train_dataset, test_dataset, params_training, params_testing, rank, ann_POD, new_path,weights)
                    
                    
                elif model == "convAE":

                    print(f"performing {model}...")
                    model_path = f"{model}.pt"
                    new_path = os.path.join(folders_path, model_path)
                    statistics.append(perform_convAE(train_dataset, test_dataset, rank, new_path))    

                elif model == "wconvAE":

                    print(f"performing {model}...")
                    model_path = f"{model}.pt"
                    new_path = os.path.join(folders_path, model_path)
                    statistics.append(perform_convAE(train_dataset, test_dataset, rank, new_path, weights))

                elif model == "NN_encoder":

                    print(f"performing {model}...")
                    ann_enc = ANN([16,64,64], nn.Tanh(), [600, 1e-12])
                    model_path = f"{model}.pt"
                    new_path = os.path.join(folders_path, model_path)
                    statistics.append(perform_NN_encoder(train_dataset, test_dataset, params_training, params_testing, rank, ann_enc, new_path))
                    
                elif model == "NN_wencoder":

                    print(f"performing {model}...")
                    ann_enc = ANN([16,64,64], nn.Tanh(), [600, 1e-12])
                    model_path = f"{model}.pt"
                    new_path = os.path.join(folders_path, model_path)
                    statistics.append(perform_NN_encoder(train_dataset, test_dataset, params_training, params_testing, rank, ann_enc, new_path, weights))

                model_stats["model"] = model
                trained_models = pd.concat(trained_models, pd.DataFrame(model_stats))
                    
            else:
                print(f"Model {model} of rank {rank} already loaded in the data frame")
    
    
    df_trained_models = pd.DataFrame(trained_models)
    # df_stat.to_csv(f'{path}/Statistics.csv', index=False)

    # with open(f'{path}/Trained_models.csv', 'w') as f:
    #     for key in trained_models.keys():
    #         f.write("%s,%s\n"%(key, trained_models[key]))
    

    
    

