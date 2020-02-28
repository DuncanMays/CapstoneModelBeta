from maxQLearning import QLearningAgent
from random import random
from numpy import arange
from matplotlib import pyplot as plt

def discretize(value, targetSet):
	closestElement = targetSet[0]
	for i in targetSet:
		if (abs(value - i) < abs(value - closestElement)):
			closestElement = i

	return i

class Agent():

	def __init__(self, actions):
		self.Q = QLearningAgent(actions)
		self.Q.startExploring()

		self.stock = 0

	def getAction(self, state):
		state.append(discretize(self.stock, arange(0, 5, 0.25)))

		action = self.Q.getAction(state)

		if action < 0:
			# if the agent buys, increase stock
			self.stock += action
		else:
			# if the agent sells, first make sure that there is enough stock
			if self.stock < action:
				action = self.stock
			
			self.stock -= action

		return action

	def giveReward(self, reward):
		self.Q.giveReward(reward)


possibleActions = [-1, -0.5, 0, 0.5, 1]

stateSet = arange(0, 5, 0.25)

numAgents = 5
agents = []
for i in range(0, numAgents):
	agents.append(Agent(possibleActions))
	# agents[i].startExploring()

prices = []

for i in range(0,1000):
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
	prices.append(actions[0])

	# calculates reward based on the actions
	for j in range(0, len(agents)):
		agents[j].giveReward(price*action)

plt.figure()
plt.plot(range(0,len(prices)), prices)
plt.show()