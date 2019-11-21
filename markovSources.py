from numpy.random import normal
import math

class MarkovSource:

	# 3 pieces of information are needed to instantiate a Markov Source, the 'schedule'
	
	def __init__(self, meanFunction, variance, initialCondition = 0):
		self.variance = variance
		self.meanFunction = meanFunction
		self.state = initialCondition

	def update(self, time):
		diff = self.state - self.meanFunction(time)
		self.state -= normal(diff/2, self.variance)
		return self.state