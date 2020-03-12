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
		# Qlearning agents can understand it. To do this, we need to know the maximum and 
		# minimum demand we can expect from the agents.
		self.maxDemand = 0
		self.minDemand = 0

		# interesting bug, if we %1, submitting 1 as the parameter
		# will result in ratio being 0, so we must %1.01, accepting the small innacuracy
		ratio = ratio%1.01
		for i in range(0, numSources):
			if (random() < ratio):
				self.sources.append(House())
			else:
				self.sources.append(Factory())

			self.maxDemand += self.sources[i].max
			self.minDemand += self.sources[i].min

		interval = (self.maxDemand - self.minDemand)/numCells
		self.demandSpace = np.arange(self.minDemand, self.maxDemand, interval)


	def update(self, time):
		self.totalDemand = 0

		for i in self.sources:
			self.totalDemand += i.update(time)

		return self.totalDemand