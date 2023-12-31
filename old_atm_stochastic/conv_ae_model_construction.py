### Model construction: FIT

"""
Module for FNN-Autoencoders.
"""

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


class convAE(Reduction, ANN):
    """
    Feed-Forward AutoEncoder class (AE)

    :param list layers_encoder: ordered list with the number of neurons of
        each hidden layer for the encoder
    :param list layers_decoder: ordered list with the number of neurons of
        each hidden layer for the decoder
    :param torch.nn.modules.activation function_encoder: activation function
        at each layer for the encoder, except for the output layer at with
        Identity is considered by default.  A single activation function can
        be passed or a list of them of length equal to the number of hidden
        layers.
    :param torch.nn.modules.activation function_decoder: activation function
        at each layer for the decoder, except for the output layer at with
        Identity is considered by default.  A single activation function can
        be passed or a list of them of length equal to the number of hidden
        layers.
    :param list stop_training: list with the maximum number of training
        iterations (int) and/or the desired tolerance on the training loss
        (float).
    :param torch.nn.Module loss: loss definition (Mean Squared if not
        given).
    :param torch.optim optimizer: the torch class implementing optimizer.
        Default value is `Adam` optimizer.
    :param float lr: the learning rate. Default is 0.001.
    :param float l2_regularization: the L2 regularization coefficient, it
        corresponds to the "weight_decay". Default is 0 (no regularization).
    :param int frequency_print: the frequency in terms of epochs of the print
        during the training of the network.
    :param boolean last_identity: Flag to specify if the last activation
        function is the identity function. In the case the user provides the
        entire list of activation functions, this attribute is ignored. Default
        value is True.

    :Example:
        >>> from ezyrb import AE
        >>> import torch
        >>> f = torch.nn.Softplus
        >>> low_dim = 5
        >>> optim = torch.optim.Adam
        >>> ae = AE([400, low_dim], [low_dim, 400], f(), f(), 2000)
        >>> # or ...
        >>> ae = AE([400, 10, 10, low_dim], [low_dim, 400], f(), f(), 1e-5,
        >>>          optimizer=optim)
        >>> ae.fit(snapshots)
        >>> reduced_snapshots = ae.reduce(snapshots)
        >>> expanded_snapshots = ae.expand(reduced_snapshots)
    """
    def __init__(self,
                 layers_encoder,
                 layers_decoder,
                 function_encoder,
                 function_decoder,
                 stop_training,
                 loss=None,
                 optimizer=torch.optim.Adam,
                 lr=0.001,
                 l2_regularization=0,
                 frequency_print=10,
                 last_identity=True):

        if layers_encoder[-1] != layers_decoder[0]:
            raise ValueError('Wrong dimension in encoder and decoder layers')

        if loss is None:
            loss = torch.nn.MSELoss()

        if not isinstance(function_encoder, list):
            # Single activation function passed
            layers = layers_encoder
            nl = len(layers)-1 if last_identity else len(layers)
            function_encoder = [function_encoder] * nl

        if not isinstance(function_decoder, list):
            # Single activation function passed
            layers = layers_decoder
            nl = len(layers)-1 if last_identity else len(layers)
            function_decoder = [function_decoder] * nl

        if not isinstance(stop_training, list):
            stop_training = [stop_training]

        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.function_encoder = function_encoder
        self.function_decoder = function_decoder
        self.loss = loss

        self.stop_training = stop_training
        self.loss_trend = []
        self.encoder = None
        self.decoder = None
        self.encoder_lin = None
        self.decoder_lin = None
        self.encoder_cnn = None
        self.decoder_cnn = None
        self.optimizer = optimizer
        self.lr = lr
        self.frequency_print = frequency_print
        self.l2_regularization = l2_regularization
 

    def _build_model(self, values):     #for us train_data
        """
        Build the torch model.

        Considering the number of neurons per layer (self.layers), a
        feed-forward NN is defined:
            - activation function from layer i>=0 to layer i+1:
              self.function[i]; activation function at the output layer:
              Identity (by default).

        :param numpy.ndarray values: the set values one wants to reduce.
        """
        # layers_encoder = self.layers_encoder.copy()
        # layers_encoder.insert(0, values.shape[1])
        #tmp = self._list_to_sequential(layers_encoder,
                                                #self.function_encoder)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 24, 2, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ELU(True),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),
            nn.BatchNorm2d(48),
            nn.ELU(True),
            nn.Conv2d(48, 96, 2, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ELU(True),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.ELU(True),
            nn.Conv2d(192, 378, 2, stride=2, padding=0),
            nn.BatchNorm2d(378),
            nn.ELU(True),
            nn.Conv2d(378, 378, 2, stride=2, padding=0),
            nn.BatchNorm2d(378),
            nn.Conv2d(378, 378, 2, stride=2, padding=0),
        )
        

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(1512, 14),
        )  
        
        self.encoder= nn.Sequential(self.encoder_cnn, self.flatten, self.encoder_lin)
            
        # layers_decoder = self.layers_decoder.copy()
        # layers_decoder.append(values.shape[1])
        self.decoder_lin = nn.Sequential(
            nn.Linear(14, 1512)
        )
           
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(378, 2, 2)),
            
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(378, 378, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(378),
            nn.ELU(True),
            nn.ConvTranspose2d(378, 378, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(378),
            nn.ELU(True),
            nn.ConvTranspose2d(378, 192, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(192),
            nn.ELU(True),
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(96),
            nn.ConvTranspose2d(96, 48, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(48),
            nn.ELU(True),
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(24),
            nn.ELU(True),
            nn.ConvTranspose2d(24, 1, 2, stride=2, padding=0, output_padding=0),

        )

        self.decoder= nn.Sequential(self.decoder_lin, self.unflatten[0], self.decoder_cnn)
        
    
    def fit(self, values, test):
        """
        Build the AE given 'values' and perform training.

        Training procedure information:
            -  optimizer: Adam's method with default parameters (see, e.g.,
               https://pytorch.org/docs/stable/optim.html);
            -  loss: self.loss (if none, the Mean Squared Loss is set by
               default).
            -  stopping criterion: the fulfillment of the requested tolerance
               on the training loss compatibly with the prescribed budget of
               training iterations (if type(self.stop_training) is list); if
               type(self.stop_training) is int or type(self.stop_training) is
               float, only the number of maximum iterations or the accuracy
               level on the training loss is considered as the stopping rule,
               respectively.

        :param numpy.ndarray values: the (training) values in the points.
        """
        # values = values.T
        self._build_model(values)

        optimizer = self.optimizer(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr, weight_decay=self.l2_regularization)

        values = self._convert_numpy_to_torch(values)
        test = self._convert_numpy_to_torch(test)
        
        n_epoch = 1
        flag = True
        while flag:
            
            y_pred = self.decoder(self.encoder(values))
            y_test = self.decoder(self.encoder(test))

            loss = self.loss(y_pred, values)
            loss_test = self.loss(y_test, test_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scalar_loss = loss.item()
            self.loss_trend.append(scalar_loss)

            for criteria in self.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if scalar_loss < criteria:
                        flag = False

            if (flag is False or
                    n_epoch == 1 or n_epoch % self.frequency_print == 0):
                print(f'[epoch {n_epoch:6d}]\t{scalar_loss:e}\t{loss_test:e}')

            n_epoch += 1

        xx = np.arange(1, n_epoch, 1)
        print(np.shape(xx), np.shape(self.loss_trend))
        plt.figure()
        plt.plot(xx, self.loss_trend, 'r')
        plt.title(f"conv_ae_{epochs}epochs loss error")
        plt.savefig(f"Error_training_convAE_{epochs}epochs_6_conv_layers.pdf", format='pdf',bbox_inches='tight',pad_inches = 0)
        plt.close()

        return optimizer

    def transform(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        X = self._convert_numpy_to_torch(X)#.T
        g = self.encoder(X)
        return g.cpu().detach().numpy().T

    def inverse_transform(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.
        """
        g = self._convert_numpy_to_torch(g).T
        u = self.decoder(g)
        return u.cpu().detach().numpy().T

    def reduce(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).

        .. note::

            Same as `transform`. Kept for backward compatibility.
        """
        return self.transform(X)

    def expand(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.

        .. note::

            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(g)
    
    def save(self, path):  

        torch.save(self, path)

        
    @staticmethod
    def load(fname):

        with open(fname, 'rb') as output:
            model = torch.load(output)   #pickle instead of torch originally

        return model 
 
### MAIN()
    
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
epochs = 2000

conv_ae = convAE([14], [14], f(), f(), epochs)  #fake structure

#Fitting
start = time()
conv_ae.fit(train_data) 
end = time() 

print("time required for the trianing", end-start)

torch.save(conv_ae, f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers.pt')

""" #conv_ae = torch.load(f'./Stochastic_results/conv_AE_{epochs}epochs_6_conv_layers.pt')

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
 """


 