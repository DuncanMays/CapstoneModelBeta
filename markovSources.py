from numpy.random import normal
import math

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

class House(MarkovSource):
	# houses have fairly high variance, but low average consumption
	# their expected consumption follows a sin wave, we should change this later
	def __init__(self, meanFunction = lambda x : 4*math.sin(x*math.pi/12) + 20, variance = 4, initialCondition = 20):
		MarkovSource.__init__(self, meanFunction, variance, initialCondition)

class Factory(MarkovSource):
	# factories have fairly low variance, but high average consumption
	def __init__(self, meanFunction = lambda x : 100, variance = 0.5, initialCondition = 100):
		MarkovSource.__init__(self, meanFunction, variance, initialCondition)

class SolarPanel(MarkovSource):
	# the mean function for solar panels looks like a lump in the middle fo the day, with relatively high variance
	def __init__(self, meanFunction = lambda x : 10/(1+((x-12)**2)/5), variance = 5, initialCondition = 0):
			MarkovSource.__init__(self, meanFunction, variance, initialCondition)

class WindTurbine(MarkovSource):
	# the mean function for wind farms is constant, with very high variance
	def __init__(self, meanFunction = lambda x : 30, variance = 8, initialCondition = 30):
			MarkovSource.__init__(self, meanFunction, variance, initialCondition)
