from maxQLearning import QLearningAgent
from random import random
from numpy import arange
from matplotlib import pyplot as plt
from copy import copy

def discretize(value, targetSet):
	closestElement = targetSet[0]
	for i in targetSet:
		if (abs(value - i) < abs(value - closestElement)):
			closestElement = i

	return closestElement

class Agent(QLearningAgent):

	def __init__(self, actions):
		self.stock = 0

		self.rewards = []
		self.stocks = []
		self.actionsTaken = []

		super(Agent, self).__init__(actions)

	# positive action indicates that the agent is selling
	# negative action indicates that the agent is buying
	def getAction(self, state):
		state = copy(state)
		state.append(discretize(self.stock, arange(0, 5, 0.25)))

		action = super(Agent, self).getAction(state)

		if action < 0:
			# if the agents buys, limit stock to 10

			if (self.stock - action > 10):
				action = self.stock - 10

			self.stock -= action

		else:
			# if the agent sells, first make sure that there is enough stock
			if self.stock < action:
				action = self.stock
			
			self.stock -= action

		self.stocks.append(self.stock)
		self.actionsTaken.append(action)

		return action

	def giveReward(self, reward):
		super(Agent, self).giveReward(reward)

		self.rewards.append(reward)


possibleActions = [-1, -0.5, 0, 0.5, 1]

stateSet = arange(0, 5, 0.25)

numAgents = 5
agents = []
for i in range(0, numAgents):
	agents.append(Agent(possibleActions))

explorationDuration = 100
activeAgentIndex = 0
activeAgent = agents[activeAgentIndex]

for i in range(0,10000):

	if (i%explorationDuration == 0):
		# switch the agent that's exploring
		activeAgent.finishExploring()
		activeAgentIndex = (activeAgentIndex+1)%len(agents)
		activeAgent = agents[activeAgentIndex]
		activeAgent.startExploring()

	supply = 5*random()
	demand = 5*random()

	Qstate = [discretize(supply, stateSet), discretize(demand, stateSet)]

	actions = []
	# gets the agent's actions
	for j in agents:
		action = j.getAction(Qstate)
		actions.append(action)

		# alters state according to the actions
		if action >= 0:
			# if the agent sold
			supply += action
		else:
			# if the agent bought
			demand += -action

	price = demand/supply

	# calculates reward based on the actions
	for j in range(0, len(agents)):
		agents[j].giveReward(price*action)

plt.figure()
toPlot = agents[0].rewards
plt.plot(range(0, len(toPlot)), toPlot)
plt.show()