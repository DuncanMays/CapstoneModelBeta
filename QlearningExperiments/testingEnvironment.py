from random import normalvariate, randint
from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt
from numpy import arange
from math import sin

# the actions that the agent will be allowed to perform
actions = ['forward', 'backward', 'stay still']

temp = arange(0, 10, 0.0001)
value = []
for i in temp:
	value.append(sin(i)+i/3)

agent = QLearningAgent(actions, exploration = 0.01)

# this list will hold the reward the agent recieves at each timestep, it will be used to plot the agent's performance
performance = []

index = 0

print('running simulation')

numIterations = 300000
for i in range(0,numIterations):
	# print(agent.lookUp([1,1,1], agent.Q))

	# resets reward
	reward = 0

	# gets action from agent
	action = agent.getAction([value[index]])

	if (action == 'forward'):
		# print('forward')
		index = (index+1)%len(value)

	elif (action == 'backward'):
		# print('backward')
		# the agent to recieves an award
		reward = value[index]
		index -= (index-1)%len(value)


	elif (action == 'stay still'):
		# print('stay still')
		pass

	else:
		print("this shouldn't happen")

	# prints progress, WILL NOT WORK WITH PYTHON2
	# erases the last output
	print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', end = '')
	# produces new output
	print(' %'+str(100*i/numIterations), end = '')

	# gives agent the reward
	agent.giveReward(reward)

	# adds reward to parformance so that it may be plotted
	performance.append(reward)

# time to calculate the average reward across a time period of num iterations
# adds new line
print()
print('plotting results')
num = 25
toPlot = []
numIterations = len(performance)-num
for i in range(0, numIterations):
	sum = 0
	for j in range(i,i+num):
		sum += performance[j]
	toPlot.append(sum/num)

	# prints progress, WILL NOT WORK WITH PYTHON2
	# erases the last output
	print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', end = '')
	# produces new output
	print(' %'+str(100*i/numIterations), end = '')

# adds new line
print()

x = range(0, len(toPlot))
plt.plot(x, toPlot)
plt.show()
