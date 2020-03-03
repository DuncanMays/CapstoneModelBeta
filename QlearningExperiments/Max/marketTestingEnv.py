from maxQLearning import QLearningAgent
from random import random
from numpy import arange
from matplotlib import pyplot as plt
from copy import copy

# quantizes a value to the closest element of a set
def discretize(value, targetSet):
	closestElement = targetSet[0]
	for i in targetSet:
		if (abs(value - i) < abs(value - closestElement)):
			closestElement = i

	return closestElement

# this is the agent that will buy and sell things on the market.
# It contains a q-learning agent that will make all the important decisions, this class
#  only exists to implement practical concerns (making sure it hasn't sold more than it
#  has, recording information fro diagnostics, etc.)
class Agent(QLearningAgent):

	def __init__(self, actions):
		# the amount of stock the agent has
		self.stock = 0

		# exist simply to evaluate performance
		self.rewards = []
		self.stocks = []
		self.actionsTaken = []

		# calls the init method of the parent class
		super(Agent, self).__init__(actions)

	# positive action indicates that the agent is selling
	# negative action indicates that the agent is buying
	def getAction(self, state):
		# since we are altering state, we must create a copy of it to avoid aliasing issues
		state = copy(state)

		# adds stock onto the state list
		state.append(discretize(self.stock, arange(0, 5, 0.25)))

		# gets the action from the qlearning class
		action = super(Agent, self).getAction(state)

		# checks to make sure the action makes sense
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

		# exist to track agent performance
		self.stocks.append(self.stock)
		self.actionsTaken.append(action)

		return action

	def giveReward(self, reward):
		super(Agent, self).giveReward(reward)

		# exist to track agent performance
		self.rewards.append(reward)


possibleActions = [-1, -0.5, 0, 0.5, 1]

# what supply and demand will be discretized to
stateSet = arange(0, 5, 0.25)

# creates the agents
numAgents = 5
agents = []
for i in range(0, numAgents):
	agents.append(Agent(possibleActions))

# the number of cylces each agent will explore before the next agent gets a turn
explorationDuration = 100
# keeps track of which agent is exploring
activeAgentIndex = 0
activeAgent = agents[activeAgentIndex]
activeAgent.startExploring()

for i in range(0,10000):

	if (i%explorationDuration == 0):
		# switch the agent that's exploring
		activeAgent.finishExploring()
		# incremements activeAgentIndex
		activeAgentIndex = (activeAgentIndex+1)%len(agents)
		activeAgent = agents[activeAgentIndex]
		activeAgent.startExploring()

	# randomly generate supply and demand
	supply = 5*random()+1
	demand = 5*random()+1

	actions = []
	# gets the agent's actions
	for j in agents:
		action = j.getAction([discretize(supply, stateSet), discretize(demand, stateSet)])
		actions.append(action)

		# alters state according to the actions
		if action >= 0:
			# if the agent sold, increase supply
			supply += action
		else:
			# if the agent bought, increase demand
			demand += -action

	# calculate price from supply and demand
	price = demand/supply

	# calculates reward based on the actions
	for j in range(0, len(agents)):
		agents[j].giveReward(price*actions[j])

plt.figure()
toPlot = agents[0].rewards
plt.plot(range(0, len(toPlot)), toPlot)
plt.show()

# prints the average reward of each agent
for i in agents:
	print(sum(i.rewards))