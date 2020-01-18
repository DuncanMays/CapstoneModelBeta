from random import normalvariate, randint
from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt

# the actions that the agent will be allowed to perform
actions = ['stop', 'go']

# the state dimensions
stateDim1 = [-1, 0.1, 1]
stateDim2 = [-1, 0.1, 1]
stateDim3 = [-1, 0.1, 1]

agent = QLearningAgent(actions)

# this list will hold the reward the agent recieves at each timestep, it will be used to plot the agent's performance
performance = []

# the state variables are randomly initialized
index1 = randint(0, len(stateDim1))-1
index2 = randint(0, len(stateDim2))-1
index3 = randint(0, len(stateDim3))-1

for i in range(0,10000):
	# print(agent.lookUp([1,1,1], agent.Q))

	# resets reward
	reward = 0

	# gets action from agent
	action = agent.getAction([stateDim1[index1], stateDim2[index2], stateDim3[index3]])

	if (action == 'go'):
		reward = stateDim1[index1]*stateDim2[index2]*stateDim3[index3]
		# the index in each dimenison is incremented upwards
		index1 = (index1+1)%len(stateDim1)
		index2 = (index2+1)%len(stateDim2)
		index3 = (index3+1)%len(stateDim3)

	elif (action == 'stop'):
		pass

	else:
		print("the agent has given something other than stop or go as its action, this shouldn't happen")

	# the state variables are randomly altered
	index1 = (index1+randint(-1,1))%len(stateDim1)
	index2 = (index2+randint(-1,1))%len(stateDim2)
	index3 = (index3+randint(-1,1))%len(stateDim3)

	# gives agent the reward
	agent.giveReward(reward)

	performance.append(reward)

# print(agent.Q)

x = range(0, len(performance))
plt.plot(x, performance)
plt.show()
