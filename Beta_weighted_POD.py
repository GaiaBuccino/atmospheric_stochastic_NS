import pandas as pd
import conv_ae
from conv_ae import convAE
import torch
from torch import nn
from scipy.stats import beta
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


#DATA LOADING
        
#data_FOM = np.load('discontinuity_snapshots.npy')
#params_FOM = np.load('disc_params.npy')
train_FOM = np.load('Data/discontinuity_training.npy')  #(180, 256, 256)
test_FOM = np.load('Data/discontinuity_testing.npy')    #(20, 256, 256)
params_training = np.load('Data/disc_params_training.npy')
params_testing = np.load('Data/disc_params_testing.npy')


a = 0.5
b = 0.5
rv = beta(a, b)

t = np.random.choice(params_training.squeeze(), 120, replace=False)
print("t =", t)

# fig, ax = plt.subplots(1, 1)
# ax.plot(x, rv.pdf((x-0.3)/2.7)/2.7, 'k-', lw=2, label='frozen pdf')
# plt.show()

weights = rv.pdf((t-0.3)/2.7)/2.7
print("weights = ", weights)

pod = POD('correlation_matrix', rank = 14)
pod.fit(train_FOM, weights)
pod_r_train = pod.transform(db_training.snapshots)