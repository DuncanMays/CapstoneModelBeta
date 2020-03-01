from numpy.random import normal
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt

hourMean = pd.read_pickle('../Data/hourmeans')

class MarkovSource:

	# 3 pieces of information are needed to instantiate a Markov Source, the meanFunction,
	# which is the 'schedule' that its expected value will follow, with markov disturbances,
	# the variance, which is the variance of the gaussian distribution that gives the markov
	# disturbance, and the initial condition, which by default is set to zero.
	def __init__(self, meanFunction, variance, initialCondition = 0):
		# allows these parameters to be accessed by other methods
		self.variance = variance
		self.meanFunction = meanFunction
		self.state = initialCondition

	# This method moves the source from one state to the next.
	# It is necessary that it takes time as a parameter, it must know the time to know what
	# its expected value should be, and it cannot determine the time on its own since the 
	# main program loop needs to be able to control intervals and such.
	def update(self, time):
		# calculates the difference between the current state and the desired expected value
		diff = self.state - self.meanFunction(time)

		# changes the state by a gaussian random variabe with variance as defined in __init__
		# and mean that's halfway between the current state and desired expected value.
		self.state -= normal(diff/2, self.variance)

		# forces the state to be positive definite
		if(self.state < 0):
			self.state = 0

		return self.state

class MarkovLive:

	'''
	series is the data that the source will emulate
	'''
	def __init__(self, series):
		self.series = series
		# calculates the standard deviation
		self.std = float(series.std())
		# initialized state
		self.x = max(series.iloc[0] + self.std*(2*np.random.random_sample()-1),0)

	def update(self, time):
		delta = (self.series.iloc[time] - self.x)/self.std
		if delta > 0:
			if np.random.random_sample() > delta:
				self.x = max(self.x + np.random.normal(loc=0,scale=delta*self.std+.1),0)
			else:
				self.x = max(self.x + np.random.normal(loc=delta*self.std/2,scale=delta*self.std+.1),0)
		else:
			if np.random.random_sample() > -delta:
				self.x = max(self.x + np.random.normal(loc=0,scale=-delta*self.std+.1),0)
			else:
				self.x = max(self.x + np.random.normal(loc=delta*self.std/2,scale=-delta*self.std+.1),0)

		return self.x

class House(MarkovLive):
	def __init__(self):
		MarkovLive.__init__(self, hourMean['Ontario']/10000)

class Factory(MarkovSource):
	# factories have fairly low variance, but high average consumption
	def __init__(self, meanFunction = lambda x : 100, variance = 1, initialCondition = 100):
		MarkovSource.__init__(self, meanFunction, variance, initialCondition)

class SolarPanel(MarkovSource):
	# the mean function for solar panels looks like a lump in the middle of the day, with relatively high variance
	def __init__(self, meanFunction = lambda x : 10/(1+((x-12)**2)/5), variance = 10, initialCondition = 0):
			MarkovSource.__init__(self, meanFunction, variance, initialCondition)

class WindTurbine(MarkovSource):
	# the mean function for wind farms is constant, with very high variance
	def __init__(self, meanFunction = lambda x : 30, variance = 16, initialCondition = 30):
			MarkovSource.__init__(self, meanFunction, variance, initialCondition)


