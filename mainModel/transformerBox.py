from markovSources import House, Factory
from random import random
from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt

class TransformerBox:

	def __init__(self, numSources, ratio):
		
		self.sources = []
		self.containsAgent = False

		# totalDemand needs to be a member of self so that an outside program can access it
		self.totalDemand = 0

		# interesting bug, if we %1, submitting 1 as the parameter
		# will result in ratio being 0, so we must %1.01, accepting the small innacuracy
		ratio = ratio%1.01
		for i in range(0, numSources):
			if (random() < ratio):
				self.sources.append(House())
			else:
				self.sources.append(Factory())


	def update(self, time):
		self.totalDemand = 0

		for i in self.sources:
			self.totalDemand += i.update(time)

		return self.totalDemand