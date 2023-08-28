import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Choose which source to use
# file_names = [ "dati-orari-Trieste molo F.lli Bandiera-%4d%02d.csv"%(year,month) for month in range(1,13) for year in [2021,2022]]
file_names = [ "wind_data/dati-orari-Boa Paloma (fino al 06_02_2018)-%4d%02d.csv"%(year,month) for month in range(1,13) for year in [2022]]


def readcsv(args):
    return pd.read_csv(args, sep=";",skipfooter=3,engine='python')

data_tot = pd.concat(map(readcsv, file_names))

# Grouping only wind speed and direction data
# modified data to test the synthetic data
ll = len(data_tot)
wind_dir = [ [ data_tot["Vento med km/h"].iloc[i], (1/4.5)*data_tot["Direzione Vento gradi N"].iloc[i] ] for i in range(ll) \
            if (not isinstance(data_tot["Vento med km/h"].iloc[i], str) and  not isinstance(data_tot["Direzione Vento gradi N"].iloc[i] , str)  ) ]


wind_dir = np.array(wind_dir)
H, speed_edges, dir_edges = np.histogram2d(wind_dir[:,0],wind_dir[:,1],bins=20)

np.save("Prob_synth.npy", H)
np.save("Speed_edges.npy", speed_edges)
np.save("Directions_edges.npy", dir_edges)

fig = plt.figure(figsize=(10,10))
plt.imshow(H.T, origin='lower',
        extent=[speed_edges[0], speed_edges[-1], dir_edges[0], dir_edges[-1]])
plt.xlabel("Speeds")
plt.ylabel("Direction")
plt.colorbar()
plt.show()

