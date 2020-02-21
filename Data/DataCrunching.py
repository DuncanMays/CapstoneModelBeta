import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import datetime
from sklearn.linear_model import LinearRegression

MasterFrame = pd.read_pickle('One_frame_to_rule_them_all')

hourmeans = MasterFrame.groupby(['Hour']).mean()
hourmeans[['Nuclear','Solar','Hydro','Wind','Gas']].plot()

#this function takes in an array of mean value data and an int defining the length of the simulation, and then computes
#a random walk of the length of the integer.
def meanwalk(series,integer):
    value = np.zeros(integer)
    std = float(series.std())
    value[0] = max(series.iloc[0] + std*(2*np.random.random_sample()-1),0)
    for i in range(integer-1):
        h = (i+1)%24
        delta = (series.iloc[h] - value[i])/std
        if delta > 0:
            if np.random.random_sample() > delta:
                value[i+1] = max(value[i] + np.random.normal(loc=0,scale=delta*std+.1),0)
            else:
                value[i+1] = max(value[i] + np.random.normal(loc=delta*std/2,scale=delta*std+.1),0)
        else:
            if np.random.random_sample() > -delta:
                value[i+1] = max(value[i] + np.random.normal(loc=0,scale=-delta*std+.1),0)
            else:
                value[i+1] = max(value[i] + np.random.normal(loc=delta*std/2,scale=-delta*std+.1),0)   
    return value

randomsolar = meanwalk(hourmeans['Solar'],24)
pd.Series(randomsolar).plot()

mean = np.array(randomsolar)
for i in range(1):
    mean = (mean+meanwalk(hourmeans['Solar'],24))/(2)
pd.Series(mean).plot()
hourmeans['Solar'].plot()
#print(mean)

hourmean = pd.read_pickle('hourmeans')
print(hourmean['Ontario'])

# def meanwalk2(series,integer):
#     value = np.zeros(integer)
#     std = float(series.std())
#     value[0] = max(series.iloc[0] + std*(np.random.random_sample()-1),0)
#     for i in range(integer-1):
#         h = (i+1)%24
#         delta = (series.iloc[h] - value[i])/std
#         if delta > 0:
#             if np.random.random_sample() > delta:
#                 value[i+1] = max(value[i] + np.random.uniform(-abs(np.sqrt(2)*delta*std),abs(np.sqrt(2)*delta*std)),0)
#             else:
#                 value[i+1] = max(value[i] + np.random.uniform(abs(delta*std/2)-abs(np.sqrt(2)*delta*std),abs(delta*std/2)+abs(np.sqrt(2)*delta*std)),0)
#         else:
#             if np.random.random_sample() > -delta:
#                 value[i+1] = max(value[i] + np.random.uniform(-abs(np.sqrt(2)*delta*std),value[i]+abs(np.sqrt(2)*delta*std)),0)
#             else:
#                 value[i+1] = max(value[i] + np.random.uniform((-abs(delta*std/2))-abs(np.sqrt(2)*delta*std),-abs(delta*std/2)+abs(np.sqrt(2)*delta*std)),0)
        
#     return value