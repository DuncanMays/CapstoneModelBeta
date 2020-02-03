from random import normalvariate, randint
from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt
from numpy import arange

# the actions that the agent will be allowed to perform
actions = ['forward', 'backward', 'stay still']

# the state dimensions
time = arange(-12, 12, 1)
localDemand = arange(-2, 2, 1)
priceOfProduction = arange(-5, 5, 1)
priceOfRetail = arange(-5, 5, 1)
charge = arange(-3, 3, 1)

agent = QLearningAgent(actions, discount = 0.25)

# this list will hold the reward the agent recieves at each timestep, it will be used to plot the agent's performance
performance = []

# the state variables are randomly initialized
# timeIndex = randint(0, len(time))-1
# localDemandIndex = randint(0, len(localDemand))-1
# priceOfProductionIndex = randint(0, len(priceOfProduction))-1
# priceOfRetailIndex = randint(0, len(priceOfRetail))-1
# chargeIndex = randint(0, len(charge))-1

timeIndex = 0
localDemandIndex = 0
priceOfProductionIndex = 0
priceOfRetailIndex = 0
chargeIndex = 0

print('running simulation')

numIterations = 500000
for i in range(0,numIterations):
	# print(agent.lookUp([1,1,1], agent.Q))

	# resets reward
	reward = 0

	# gets action from agent
	action = agent.getAction([time[timeIndex], localDemand[localDemandIndex], priceOfProduction[priceOfProductionIndex], priceOfRetail[priceOfRetailIndex], charge[chargeIndex]])

	if (action == 'forward'):
		# print('forward')
		# the index in each dimenison is incremented upwards
		timeIndex = (timeIndex+1)%len(time)
		localDemandIndex = (localDemandIndex+1)%len(localDemand)
		priceOfProductionIndex = (priceOfProductionIndex+1)%len(priceOfProduction)
		priceOfRetailIndex = (priceOfRetailIndex+1)%len(priceOfRetail)
		chargeIndex = (chargeIndex+1)%len(charge)

	elif (action == 'backward'):
		# print('backward')
		# the agent to recieves an award
		reward = time[timeIndex] + localDemand[localDemandIndex] + priceOfProduction[priceOfProductionIndex] + priceOfRetail[priceOfRetailIndex] + charge[chargeIndex]

		# the index in each dimenison is incremented downwards, with no reward, this will test if the agent can think ahead
		timeIndex = (timeIndex-1)%len(time)
		localDemandIndex = (localDemandIndex-1)%len(localDemand)
		priceOfProductionIndex = (priceOfProductionIndex-1)%len(priceOfProduction)
		priceOfRetailIndex = (priceOfRetailIndex-1)%len(priceOfRetail)
		chargeIndex = (chargeIndex-1)%len(charge)

	elif (action == 'stay still'):
		# print('stay still')
		pass

	else:
		print("this shouldn't happen")

	# the state variables are randomly altered
	# timeIndex = (timeIndex+randint(-1,1))%len(time)
	# localDemandIndex = (localDemandIndex+randint(-1,1))%len(localDemand)
	# priceOfProductionIndex = (priceOfProductionIndex+randint(-1,1))%len(priceOfProduction)
	# priceOfRetailIndex = (priceOfRetailIndex+randint(-1,1))%len(priceOfRetail)
	# chargeIndex = (chargeIndex+randint(-1,1))%len(charge)

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
num = 50
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

x = range(0, len(toPlot[200000:400000]))
plt.plot(x, toPlot[200000:400000])
plt.show()
