# ### Evaluating errors
"""
Module for FNN-Autoencoders

"""
import pandas as pd
import conv_ae
import pod
from rom import ReducedOrderModel as ROM

from conv_ae import convAE
import torch
from torch import nn
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from ezyrb import Database, ANN, AE, RBF, POD
#from ezyrb import ReducedOrderModel as ROM
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

def compute_expected_value(dataset: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray :

    weighted_dataset = np.zeros((dataset.shape[0],dataset.shape[1]))
    
    if weights is not None:
        
        for kk in range(dataset.shape[0]):
            weighted_dataset[kk][:] = dataset[kk][:]*weights[kk]
        E_value = np.sum(weighted_dataset, axis=0)/np.sum(weights)

    else:
        E_value = np.mean(dataset, axis=0)

    return E_value
    
def compute_variance(dataset: np.ndarray, avg_sol: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> np.ndarray:   # dataset = (180, 65k)
    """_summary_

    Args:
        dataset (np.ndarray): 1000 x 65536
        avg_sol (Optional[np.ndarray], optional): _description_. Defaults to None.
        weights (Optional[np.ndarray], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    if avg_sol.any() == None:
        avg_sol = compute_expected_value(dataset)

    var_dataset = np.zeros((dataset.shape[0], dataset.shape[1]))

    if weights is not None:
        var_dataset[:,:] = (dataset-avg_sol)**2 * weights.reshape((-1,1))

        # for kk in range(dataset.shape[0]):
        #     var_dataset[kk,:] = (dataset[kk,:] - avg_sol)**2*weights[kk]

        var = np.sum(var_dataset, axis=0)/np.sum(weights)

    else:

        var_dataset[:,:] = (dataset-avg_sol)**2 
        var = np.sum(var_dataset, axis=0)/(dataset.shape[0]) 

    return var

def compute_stats(dataset: np.ndarray, trained_model: str, model: str, E_FOM:np.ndarray, var_FOM:np.ndarray, params: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> dict:
    #compute_stats(dataset, model_to_be_loaded, POD14, weights) 
    # if model == 'POD':
    #     exp_model = pod.POD()
    # #understand how to load
    # if model=='wPOD'or model == 'POD_NN' or model == 'wPOD_NN' or model == 'NN_encoder' or model == 'NN_wencoder': #"POD", "wPOD"]#, "POD_NN", "wPOD_NN", "convAE", "wconvAE", "NN_encoder", "NN_wencoder"
        
    #     exp_model = ROM.load(f"{trained_model['model_path']}")
        
    #     if model=='POD' or model=='wPOD':
    #         exp_model = exp_model.reduction

    # elif model == 'convAE' or model == 'wconvAE' or model == 'NN_encoder' or model == 'NN_wencoder':

    #     exp_model = torch.load(trained_model['model_path'])

    exp_model = ROM.load(trained_model)
    sol = compute_approx(dataset, params, exp_model, model)
    E_value = compute_expected_value(sol, weights)
    var = compute_variance(sol, E_value, weights)

    error_E = np.linalg.norm(E_value - E_FOM)/np.sqrt(E_value.shape[0])
    error_var = np.linalg.norm(var - var_FOM)/np.sqrt(E_value.shape[0])

    res_dict = {'model': f'{model}',
                'expected value': E_value,
                'variance': var,
                'error expected value wrt FOM': error_E,
                'error variance wrt FOM': error_var
                }
        
    return res_dict

def compute_approx(dataset:np.ndarray, params: np.ndarray, method: ROM, model: str) -> np.ndarray:

    dataset_line = dataset.reshape(len(dataset), -1)

    if 'NN' not in f'{model}':        

        if 'convAE' in f'{model}':
            # convAE and wconvAE

            tensor = np.expand_dims(dataset, axis=1)     
            
            pred = method.reduction.inverse_transform(\
                method.reduction.transform(tensor)).T
            pred = pred.reshape(len(pred), -1)
            
        else:
            # POD and wPOD

            pred = method.reduction.inverse_transform(\
                method.reduction.transform(dataset_line.T)).T
            
    ### reduction and approximation ###

    else:
        
        if 'encoder' in f'{model}':
            # encoder-NN and wencoder-NN
            reduced_coefficients = method.approximation.predict(params)
            pred = method.reduction.inverse_transform(reduced_coefficients.T).T

            pred = pred.reshape(len(pred), -1)

        else:
            # POD-NN and wPOD-NN
            pred = np.zeros((dataset_line.shape[0],dataset_line.shape[1]))
            for ii in range(len(params)):

                pred[ii,:] = method.predict(params[ii])


    return pred

def perform_method(method:str, db_train:Database, rank:int,  dump_path:str, \
                reduction:pod.POD or convAE, approximation:Optional[ANN] = None, \
                weights:Optional[np.ndarray] = None ,  fit_only_approximation:Optional[bool] = False)\
                -> Tuple[np.ndarray, np.ndarray, dict]:  
    """_summary_

    Args:
        method (str): _description_
        train_dataset (np.ndarray): _description_
        test_dataset (np.ndarray): _description_
        params_training (np.ndarray): _description_
        params_testing (np.ndarray): _description_
        rank (int): _description_
        dump_path (str): _description_
        reduction (pod.PODorconvAE): _description_
        approximation (Optional[ANN], optional): _description_. Defaults to None.
        weights (Optional[np.ndarray], optional): _description_. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: _description_
    """
    ### data reshaping ###

    # train_dataset_line = train_dataset.reshape(len(train_dataset), -1)
    # db_train = Database(params_training, train_dataset_line)

    ### fit ###
    if approximation == None:
        approximation = ANN([16,64,64], nn.Tanh(), [400, 1e-12])

    Rom = ROM(db_train, reduction, approximation, weights)
    train_time = -perf_counter()
    if fit_only_approximation:
        reduced_output = Rom.reduction.transform(Rom.database.snapshots.T).T

        if Rom.scaler_red:
            reduced_output = Rom.scaler_red.fit_transform(reduced_output)

        Rom.approximation.fit(Rom.database.parameters,
            reduced_output)
    else:
        Rom.fit(weights)
    train_time += perf_counter()
    Rom.save(f'{dump_path}',save_db=False)

    return Rom, train_time


def compute_prediction_errors(Rom:ROM, method:str, train_dataset:np.ndarray, \
                test_dataset:np.ndarray, params_training:np.ndarray, \
                params_testing:np.ndarray, rank:int,  dump_path:str )\
                -> Tuple[np.ndarray, np.ndarray, dict]:  

    ### data reshaping ###

    train_dataset_line = train_dataset.reshape(len(train_dataset), -1)
    test_dataset_line = test_dataset.reshape(len(test_dataset), -1)



    ### simple reduction ###
    pred_train = compute_approx(train_dataset, params_training, Rom, method)
    pred_test  = compute_approx(test_dataset, params_testing, Rom, method)

    pred_train = pred_train.reshape(len(train_dataset), -1)
    pred_test = pred_test.reshape(len(test_dataset), -1)

    error_train = np.zeros(len(train_dataset_line))
    error_test = np.zeros(len(test_dataset_line))

    for kk in range(len(train_dataset_line)):
        error_train[kk] = np.linalg.norm(pred_train[kk] - train_dataset_line[kk])/np.linalg.norm(train_dataset_line[kk])

    for jj in range(len(test_dataset_line)):
        error_test[jj] = np.linalg.norm(pred_test[jj] - test_dataset_line[jj])/np.linalg.norm(test_dataset_line[jj])

    statistics = {#'model name': method + f'{rank}',
                  'method': method,
                  'rank': rank,
                  'training error': error_train,
                  'testing error': error_test,
                  'model path': dump_path} 



    return pred_train, pred_test, statistics

### main ###

types_test = ["synthetic_discontinuity"]#, "synthetic_discontinuity_stats"]  #"simulated_gulf"]  #"snapshots" or "discontinuity"      "synthetic_discontinuity","synthetic_discontinuity", 

for test in types_test:

    train_dataset, test_dataset, params_training, params_testing = prepare_data(test, 'Data')

    train_dataset_line = train_dataset.reshape(len(train_dataset), -1)
    db_train = Database(params_training, train_dataset_line)

    #weights = np.ones(len(params_training))
    #weights = np.load("weights.npy")
    weights = beta.pdf((params_training - 0.3) / 2.7, 5, 2).squeeze() / 2.7

    path = f"./Stochastic_results/synthetic_discontinuity_tests/"
  
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(error) 

    ranks = [14]    # 2, 6, 14, 30


    methods = ["POD","wPOD", "POD_NN", "wPOD_NN","convAE", "wconvAE", "NN_encoder", "NN_wencoder"]
    #ok: "convAE", "wconvAE","POD", "wPOD", "POD_NN", "wPOD_NN"
    #statistics = {}
    model_infos = {}
    stats_models = {}
    leg = []


    new_paths = dict()
    for method in methods:
        for rank in ranks:
            model = method + f'_{rank}'
            directory = f"{model}" 
            folders_path = os.path.join(path, directory)

            if method == "POD":
                new_paths[model]  = os.path.join(folders_path, f"{method}.rom")
            elif method== "wPOD":
                new_paths[model]  = os.path.join(folders_path, f"{method}.rom")
            elif method== "POD_NN":
                epochs = 25000
                new_paths[model] = os.path.join(folders_path, f"{method}_{epochs}ep.rom")
            elif method== "wPOD_NN":
                epochs = 25000
                new_paths[model] = os.path.join(folders_path, f"{method}_{epochs}ep.rom")
            elif method== "convAE":
                epochs = 8000
                new_paths[model]  = os.path.join(folders_path, f"{method}_{epochs}ep.rom")
            elif method== "wconvAE":
                epochs = 8000
                new_paths[model] = os.path.join(folders_path, f"{method}_{epochs}ep.rom")
            elif method== "NN_encoder":
                cAE_epochs = 8000
                NN_epochs = 25000
                new_paths[model] = os.path.join(folders_path, f"{method}_{cAE_epochs}ep_cAE_{NN_epochs}ep_NN.rom")
            elif method== "NN_wencoder":
                cAE_epochs = 8000
                NN_epochs = 25000
                new_paths[model] = os.path.join(folders_path, f"{method}_{cAE_epochs}ep_cAE_{NN_epochs}ep_NN.rom")





    if 'stats' not in f'{test}':

        if os.path.exists(f'{path}/Trained_models.pkl'):
            with open(f'{path}/Trained_models.pkl', 'rb') as fp:
                model_infos = pickle.load(fp)
        else:
            model_infos = dict()

        fig, ax = plt.subplots()
                           
        for method in methods:          
        

            for rank in ranks:

                model = method + f'_{rank}'

                directory = f"{model}" 
                folders_path = os.path.join(path, directory)
                try:
                    os.makedirs(folders_path, exist_ok=True)
                except OSError as error:
                    print(error) 

                
                Rom=ROM.load(new_paths[model])

                Rom.save(new_paths[model],save_db=False)
                
