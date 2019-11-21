from matplotlib import pyplot as plt
from numpy import arange
from numpy.random import normal
from markovSources import MarkovSource
import math

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes
T = arange(0, 24, 0.25)

# baseline power production, this will be the amount of electricity produced by nuclear
# power plants
baseline = 9779

# I am going to make the hydro production schedule a sin wave just to start out with,
# we should definetely look at IESO data and find a function that fits it better 
hydroSchedule = lambda x : 1900*math.sin(x*math.pi/12) + 1900

# main program loop
# each iteration of this loop represents one day in the model
# while(True):
# for dummy in range(0,100):
# 	# each iteration in this loop represents one time interval
# 	for t in T:
# 		totalProduction = 0
# 		totalDemand = 0

# 		# production from nuclear and hydro plants is added to the grid
# 		totalProduction += baseline
# 		totalProduction += hydroSchedule(t)

# 		# both stochastic consumers and producers determine the amount they consume and 
# 		# produce respectively

# 		# production from gas-fired plants is added to the grid