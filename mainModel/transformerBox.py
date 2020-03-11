from markovSources import House, Factory
from random import random
from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt
import numpy as np

class TransformerBox:

	def __init__(self, numSources, ratio, numCells = 5):
		
		self.sources = []
		self.containsAgent = False

		# totalDemand needs to be a member of self so that an outside program can access it
		self.totalDemand = 0

		# the transformer box needs to quantize the electrical demand moving through it so that
		# Qlearning agents can understand it. To do this, we need to know the maximum demand, as
		# well as have some kind of interval to quantize things with.
		demandInterval = 10
		maxDemand = 0

		# interesting bug, if we %1, submitting 1 as the parameter
		# will result in ratio being 0, so we must %1.01, accepting the small innacuracy
		ratio = ratio%1.01
		for i in range(0, numSources):
			if (random() < ratio):
				self.sources.append(House())
				maxDemand += self.sources[i].max
			else:
				self.sources.append(Factory())
				maxDemand += self.sources[i].max

		self.demandSpace = np.arange(0, maxDemand, demandInterval)


	def update(self, time):
		self.totalDemand = 0

		for i in self.sources:
			self.totalDemand += i.update(time)

		return self.totalDemand