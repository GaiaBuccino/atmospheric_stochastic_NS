import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directions_synth = np.load("Directions_edges.npy")
speed_synth = np.load("Speed_edges.npy")
prob_synth = np.load("Prob_synth.npy")  #H

# choose of a speed value 2
####
fixed_speed = 2
H_speed_fix = np.sum(prob_synth[fixed_speed][:])
print("sum H_speed2 = ", H_speed_fix)

####
prob_synth = prob_synth/H_speed_fix

dirs = np.linspace(np.min(directions_synth) ,np.max(directions_synth), 300) #necessaria?
samples = 180
directions_train= np.random.choice(dirs, samples, replace=False)

print("directions synth = ", directions_synth)


weights = np.zeros(samples)
directions_train.sort()
print("directions train = ", directions_train)

i=0
count = directions_synth.size-1
for k in range(directions_train.size):
    flag = False
    while (flag!=True and i in range(count)):

        if directions_train[k] >= directions_synth[i] and directions_train[k] < directions_synth[i+1]:
            flag = True
            weights[k] = prob_synth[fixed_speed][i]
            
        else: 
            i = i+1

print(weights)
print("somma weights = ", np.sum(prob_synth[fixed_speed][:]))
# print(prob_synth[5][2])
np.save("weights.npy", weights)







