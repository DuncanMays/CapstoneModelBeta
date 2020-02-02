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

agent = QLearningAgent(actions)

# this list will hold the reward the agent recieves at each timestep, it will be used to plot the agent's performance
performance = []

# the state variables are randomly initialized
timeIndex = randint(0, len(time))-1
localDemandIndex = randint(0, len(localDemand))-1
priceOfProductionIndex = randint(0, len(priceOfProduction))-1
priceOfRetailIndex = randint(0, len(priceOfRetail))-1
chargeIndex = randint(0, len(charge))-1

for i in range(0,500000):
	# print(agent.lookUp([1,1,1], agent.Q))

	# resets reward
	reward = 0

	# gets action from agent
	action = agent.getAction([time[timeIndex], localDemand[localDemandIndex], priceOfProduction[priceOfProductionIndex], priceOfRetail[priceOfRetailIndex], charge[chargeIndex]])

	if (action == 'forward'):
		# the index in each dimenison is incremented upwards
		timeIndex = (timeIndex+1)%len(time)
		localDemandIndex = (localDemandIndex+1)%len(localDemand)
		priceOfProductionIndex = (priceOfProductionIndex+1)%len(priceOfProduction)
		priceOfRetailIndex = (priceOfRetailIndex+1)%len(priceOfRetail)
		chargeIndex = (chargeIndex+1)%len(charge)

	elif (action == 'backward'):
		# the agent to recieves an award
		reward = time[timeIndex] + localDemand[localDemandIndex] + priceOfProduction[priceOfProductionIndex] + priceOfRetail[priceOfRetailIndex] + charge[chargeIndex]

		# the index in each dimenison is incremented downwards, with no reward, this will test if the agent can think ahead
		timeIndex = (timeIndex-1)%len(time)
		localDemandIndex = (localDemandIndex-1)%len(localDemand)
		priceOfProductionIndex = (priceOfProductionIndex-1)%len(priceOfProduction)
		priceOfRetailIndex = (priceOfRetailIndex-1)%len(priceOfRetail)
		chargeIndex = (chargeIndex-1)%len(charge)

	elif (action == 'stay still'):
		pass

	else:
		print("the agent has given something other than stop or go as its action, this shouldn't happen")

	# the state variables are randomly altered
	timeIndex = (timeIndex+randint(-1,1))%len(time)
	localDemandIndex = (localDemandIndex+randint(-1,1))%len(localDemand)
	priceOfProductionIndex = (priceOfProductionIndex+randint(-1,1))%len(priceOfProduction)
	priceOfRetailIndex = (priceOfRetailIndex+randint(-1,1))%len(priceOfRetail)
	chargeIndex = (chargeIndex+randint(-1,1))%len(charge)

	# gives agent the reward
	agent.giveReward(reward)

	# adds reward to parformance so that it may be plotted
	performance.append(reward)

# time to calculate the average reward across a time period of num iterations
print('plotting')
num = 100
toPlot = []
for i in range(0, len(performance)-num):
	sum = 0
	for j in range(i,i+num):
		sum += performance[j]
	# print(performance[i])
	toPlot.append(sum)

x = range(0, len(toPlot))
plt.plot(x, toPlot)
plt.show()
