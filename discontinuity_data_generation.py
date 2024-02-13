#Evaluating discontinuity errors

import numpy as np
import os
from time import time


my_path = os.path.abspath(__file__)
try:
    os.mkdir("Data")
except:
    print("Folder already exists")

tests = ["synthetic_discontinuity","synthetic_discontinuity_stats"]

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

