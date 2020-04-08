print('importing required modules')

from matplotlib import pyplot as plt
from numpy import arange
from markovSources import House, SolarPanel, WindTurbine
from transformerBox import TransformerBox
import math
from random import randint, choice
from QLearningAgent import QLearningAgent
from copy import copy
import pandas as pd

print('setting up')

discountFactor = 0.25

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes, as we may want to change 
# this as our Q-learning algorithm evolves.
# numpy.arange does the same thing as linspace in Matlab
timestep = 2
T = arange(0, 24, timestep)

# the number of values that local demand can take, for the Qlearning agents
localDemandCells = 5
# local demand space will be given by the individual boxes

# I decided to provide a quantization for global demand in the event we implement post-pricing,
# most of the intellectual work had to be done to calculate nuclear output anyway.
globalDemandCells = 5
globalDemandSpace = []

# the number of values that the retail price of electricity can take, for the Qlearning agents
retailPriceCells = 5
retailPriceSpace = []

# the number of values that the production price of electricity can take, for the Qlearning agents
prodPriceCells = 7
prodPriceSpace = []

actionSpace = [-70, 0, 70]
# this is the first example of a pattern that repeats itself a bunch of times,
# we determine a maximum vlaue a quantity can take, and a minimum quatnity. and then we form a 
# quantization with a given number of cells in that range. In this case we are quantizint the
# battery's charge. This needs to happen before we define the batter class below.
minCharge = 0
maxCharge = 140
chargeSpace = [0, 70, 140]

demandData = pd.read_pickle('../Data/2018Demand')
# the factor of 30 makes it at least somewhat resemble previous sims
demandData = demandData['Ontario']*30

# this method takes a value and returns the closest element of the set it is given
def quantize(value, targetSet):
	closestElement = targetSet[0]
	for i in targetSet:
		if (abs(value - i) < abs(value - closestElement)):
			closestElement = i

	return closestElement

def deepCopy(obj):
	obj = copy(obj)
	if isinstance(obj, dict):
		for i in obj:
			obj[i] = deepCopy(obj[i])
	
	return obj


# this class represents batterys in our model
class Battery(QLearningAgent):

	def __init__(self, actionSpace, transformerBox):
		self.charge = 0
		self.transformerBox = transformerBox

		# this array will be used to track the rewards of the agent, for performance evaluation
		self.rewards = []
		self.actions = []

		# calling the init method of the parent class
		super(Battery, self).__init__(actions=actionSpace, discount = discountFactor)

	def chooseAction(self, priceOfProduction, priceOfRetail, time):
		# quantizing local demand, pulling totalDemand from the box will only work if get action
		# is called after update is called on the box.
		# localDemand = quantize(self.transformerBox.totalDemand, self.transformerBox.demandSpace)

		charge = quantize(self.charge, chargeSpace)

		prodPrice = quantize(priceOfProduction, prodPriceSpace)

		retailPrice = quantize(priceOfRetail, retailPriceSpace)

		time = quantize(time, T)

		state = [time, prodPrice, priceOfRetail, charge]

		action = super(Battery, self).getAction(state)

		# we now must check to make sure that the action makes sense
		# if we sell, have we sold more than the battery's charge?
		# if we buy, will we reach the battery's capacity?

		# negative action signifies selling electricity
		if (action < 0):
			# the agent sold, we must check that we did not sell more that we have charged
			if (-action > self.charge):
				action = -self.charge
		elif (action > 0):
			# the agent bought, we must check that we did not buy more than we have capacity for
			if (action + self.charge > maxCharge):
				action = maxCharge - self.charge

		self.charge += action

		return action

	# this stops the agent from learning any more, effectively freezing their policy
	def freeze(self):
		self.exploration = 0
		self.changeAlpha(self.Q, 0.000000001)

	def rejuvenate(self):
		self.exploration = 0.1
		self.changeAlpha(self.Q, 1)

	def changeAlpha(self, q, alpha):
		if(('type' in q) and (q['type'] == 'leaf')):
			# this block will execute if q is a leaf
			# iterates accross the actions to change all the alpha values
			for i in self.actions:
				q[i]['alpha'] = alpha

		else:
			# this block will execute if q is a node in the Qtree
			for i in q:
				# recursively calls changeAlpha on all branches off of q
				self.changeAlpha(q[i], alpha)


# doesn't really matter what hydroTarget is initialized it, it will be updated to
#  something usefull on the first cyle of the model. So long as whatever it is has the 
#  same number of elements as T, the model will work. For this reason i've initialized
#  it to T
hydroTarget = {T[i]: 0 for i in range(0, len(T))}

# the price of production for various methods in dollars per dollars / (megawatt/Hours)^2
nuclearPrice = 0.00032
hydroPrice = 0.00790
# gasPrice = 0.00975 + 5.29
gasPrice = 0.00975
windPrice = -0.00130
solarPrice = -0.02063

# we will need these to create a quantization for the production price of electricity
prodCostMin = 0
prodCostMax = 0

solarPanels = []
numSolarPanels = 20
# creates numSolarPanels instances of SolarPanel
for i in range(0,numSolarPanels):
	solarPanels.append(SolarPanel())
	prodCostMin += solarPrice*solarPanels[i].min
	prodCostMax += solarPrice*solarPanels[i].max

WindTurbines = []
numWindTurbines = 50
# creates numWindTurbines instances of WindTurbine
for i in range(0,numWindTurbines):
	WindTurbines.append(WindTurbine())
	prodCostMin += windPrice*WindTurbines[i].min
	prodCostMax += windPrice*WindTurbines[i].max

# prodPrice min and max will take hydro power into account when the hydro schedule is set below in the loop

# creates the power boxes through which demand will flow and a Qlearning agent will take action
numBoxes = 500
boxes = []
for i in range(0, numBoxes):
	# the first parameter is the number of total demand agents the box will service
	# the second parameter is the ratio between the number of houses and the number of factories served.
	boxes.append(TransformerBox(randint(20, 50), 0.9, numCells = localDemandCells))

# we will now attach batteries to some of the boxes
numBatteries = 200
batteries = []
# we need recursive behavior, so I will write this as a function
# doing it like this means that the program will crash if the number of batterys exceeds the number of boxes
def assignBattery(box):
	if (box.containsAgent):
		# the box already has a battery, meaning we must try again
		assignBattery(choice(boxes))
	else:
		# the box has no battery
		# actions are -1 to 1 at intervals of 1, this almost certainly will need to change
		box.containsAgent = True
		battery = Battery(actionSpace, box)
		batteries.append(battery)

for i in range(0, numBatteries):
	# assigns a batter to a random box
	assignBattery(choice(boxes))

# we now create the space in which global demand will live in, we will not
# use it right now but we very well might when we implement post-procing.
# we do this by first determining the range that demand can take values in
minGlobalDemand = sum([boxes[i].minLocalDemand for i in range(0, len(boxes))])
maxGlobalDemand = sum([boxes[i].maxLocalDemand for i in range(0, len(boxes))])

# minGlobalDemand = min(demandData)
# maxGlobalDemand = max(demandData)

# we then quantize this range by the number of demand cells we want
interval = (minGlobalDemand - maxGlobalDemand)/globalDemandCells
globalDemandSpace = arange(minGlobalDemand, maxGlobalDemand, interval)

# baseline power production, this will be the amount of electricity produced by nuclear
# power plants
baseline = minGlobalDemand

prodCostMin += nuclearPrice*baseline
prodCostMax += nuclearPrice*baseline

# to determine min and max production values for hydro power we need min and max global power,
# as well as the min and max production for each source
maxGlobalProd = baseline + sum(solarPanels[i].max for i in range(0,len(solarPanels))) + sum(WindTurbines[i].max for i in range(0,len(WindTurbines)))
minGlobalProd = baseline + sum(solarPanels[i].min for i in range(0,len(solarPanels))) + sum(WindTurbines[i].min for i in range(0,len(WindTurbines)))

# if I was really concerned with this, I'd make it max - min, but i am not very concerned with
# this as the maxima and minima of sources usually happen at the same time, and doing so might
# cause the quantization to be a poor approximation.
maxHydro = 0
# really just minGlobalDemand - baseline, but baseline = minGlobalDemand
minHydro = 0

prodCostMin += hydroPrice*minHydro
prodCostMax += hydroPrice*maxHydro

# we must normalize
prodCostMax = prodCostMax/maxGlobalProd
prodCostMin = prodCostMin/minGlobalProd

interval = (prodCostMax - prodCostMin)/prodPriceCells
prodPriceSpace = arange(prodCostMin, prodCostMax, interval)

# this is something that needs to change to a function that at least somewhat mimics reality
def retailprice(time):
    if 0<= time < 7:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0065/3
    elif 7 <= time < 11:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0094/3
    elif 11 <= time < 17:
        #On peak rates, 13.4¢/kWh, 13.4¢ / 100(¢/$) * 1000 kWh/MWh = 134$/MWh
        # priceOfRetail = 134
        priceOfRetail = 0.0134/3
    elif 17 <= time < 19:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0094/3
    elif 19 <= time <= 24:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0065/3
    
    return priceOfRetail

retailPriceSpace = [134, 94, 65]
retailPriceSpace = [0.0134/3, 0.0094/3, 0.0065/3]

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []
retailPrice = []
windProd = []
solarProd = []
hydroProd = []
gasProd = []
avgActions = []
avgReward = []

print('starting model')

oldQ = {}

# main program loop
# each iteration of this loop represents one day in the model
for day in range(0, 10):
	print('day: '+str(day))

	# hydro power will try to match the power defecit of the day before, so while
	#  hydroSchedule is read from in the loop below, hydroTarget will be written to
	hydroSchedule = hydroTarget

	# clears hydroTarget for the next day
	hydroTarget = {}

	# each iteration in this loop represents one time interval (not a full day)
	for t in T:
		# resets these variables to zero
		totalProduction = 0
		totalDemand = 0
		totalCostOfProd = 0
		priceOfRetail = retailprice(t)

		# production  from uncontrolled sources must be added to the grid first, 
		#  as all other variables (demand, production from controlled sources like 
		#  hydro and gas, as well as discharge from batteries) are determined using it.

		# production from nuclear plants is added to the grid, 
		# with necessary adjustments to totalCostOfProd
		totalProduction += baseline
		totalCostOfProd += nuclearPrice*baseline

		# production from solar panels is added to the grid
		for panel in solarPanels:
			temp = panel.update(t)
			totalProduction += temp
			totalCostOfProd += solarPrice*temp
		solarProd.append(totalProduction)

		# production from wind turbines is also added to the grid
		for turbine in WindTurbines:
			temp = turbine.update(t)
			totalProduction += temp
			totalCostOfProd += windPrice*temp
		windProd.append(totalProduction)

		# consumption must happen before any Q learning agents take action, and
		#  before hydro and gas, as this will determine the electricity defecit,
		#  which hydro will make up for in the next day and gas will make up for
		#  today is one of the parameters of the learning agents. This might get
		#  tricky if we ever implement post-pricing, as demand will be determined
		#  by price.

		#  users draw electricity from grid
		for i in boxes:
			totalDemand += i.update(t)

		# we need to modulate the time, since
		# time = 24*day + t
		# index = time%len(demandData)
		# totalDemand = demandData[index]

		# hydro power matches the difference between totalDemand and nuclear power and stochastic producers
		# and so we must record that difference here
		diff = totalDemand - totalProduction
		if (diff < 0):
			# checks to make sure hydro production cant be negative
			diff = 0
		hydroTarget[t] = diff

		# adds hydro power to the grid
		totalProduction += hydroSchedule[t]
		totalCostOfProd += hydroPrice*hydroSchedule[t]
		hydroProd.append(totalProduction)

		# the price of electricity is increased as if gas production has taken place, this means that agents will buy and 
		#  sell at the price of electricity they would pay with gas power, but gas will only be added to the grid after 
		#  the agents act.
		hypotheticalGasProduction = 0
		if(totalDemand > totalProduction):
			hypotheticalGasProduction = totalDemand - totalProduction
			totalCostOfProd += gasPrice*hypotheticalGasProduction

		# right now, price of production is the total price to produce all the electricity in the system, so we must divide
		# it by the amount of electricity in the system to get the price per MWh
		priceOfProd = totalCostOfProd/totalProduction 

		actionSum = 0
		rewardSum = 0

		if (day != 0):
			# get agents actions from time, price of retail, price of production
			# the local demand, as well as the battery's capacity, will be added to state within the battery class.
			for j in batteries:
				action = j.chooseAction(priceOfProd, priceOfRetail, t)
				reward = 0
				if action > 0:
					# if the agent bought
					totalDemand += action
					reward = -priceOfProd*action
				elif action < 0:
					# if the agent sold
					totalProduction -= action
					reward = -priceOfRetail*action

				j.giveReward(reward)

				actionSum += action
				rewardSum += reward

			avgActions.append(actionSum/len(batteries))
			avgReward.append(rewardSum/len(batteries))

		# we only add gasProduction after the Q learning agents have done their thing
		gasProduction = 0
		if(totalDemand > totalProduction):
			gasProduction = totalDemand - totalProduction
		totalProduction += gasProduction

		# these lines only serve to make plots below
		gasProd.append(totalProduction)
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(priceOfProd)
		retailPrice.append(priceOfRetail*30000)

	if(day == 0):

		# finds the max and min hydro production values
		maxHydro = hydroTarget[0]
		minHydro = hydroTarget[0]
		for i in hydroTarget:
			if (hydroTarget[i] > maxHydro):
				maxHydro = hydroTarget[i]

			elif (hydroTarget[i] < minHydro):
				minHydro = hydroTarget[i]

		maxGlobalProd += maxHydro
		minGlobalProd += minHydro

		# GUESTIMATION
		maxGlobalProd += 50000
		minGlobalProd -= 50000


		prodCostMax += hydroPrice*maxHydro
		prodCostMin += hydroPrice*minHydro

		# GUESTIMATION
		prodPriceMax = prodCostMax/minGlobalProd
		prodPriceMin = prodCostMin/maxGlobalProd - 0.001

		# sets up the quantization for prodPrice
		interval = (prodPriceMax - prodPriceMin)/prodPriceCells
		prodPriceSpace = arange(prodPriceMin, prodPriceMax, interval)

	# if (day == 3000):
	# 	print('freezing all agents but one')

	# 	batteries[0].rejuvenate()

	# 	oldQ = deepCopy(batteries[0].Q)

	# 	for i in range(1, len(batteries)):
	# 		batteries[i].freeze()

# the first day is tainted data as it is a callibration day, we will cut it out of the data
demand = demand[len(T):len(demand)]
production = production[len(T):len(production)]
prodPrice = prodPrice[len(T):len(prodPrice)]
retailPrice = retailPrice[len(T):len(retailPrice)]

# plots everything all nice and pretty
x = range(0, len(solarProd))
# plt.plot(x, demand, label='demand')
# plt.plot(x, [maxGlobalDemand for i in range(0, len(x))])
# plt.plot(x, [minGlobalDemand for i in range(0, len(x))])
# plt.plot(x, production, label='production')
# plt.plot(x, [maxGlobalProd for i in range(0, len(x))])
# plt.plot(x, [minGlobalProd for i in range(0, len(x))])
# plt.plot(x, prodPrice, label='prodprice')
# plt.plot(x, [prodPriceMax for i in range(0, len(x))])
# plt.plot(x, [prodPriceMin for i in range(0, len(x))])
# plt.plot(x[95*4:100*4], retailPrice[95*4:100*4], label='retailPrice')
# plt.plot(x, batteries[0].toPlot)
# plt.plot(x[:len(x)-12][len(avgActions)-12*5:], avgActions[len(avgActions)-12*5:])
# plt.plot(x[95*4:100*4], gasProd[95*4:100*4], label='gas production')
# plt.plot(x[0:12], [baseline for i in range(0, len(x))][len(x)-12*5:len(x)-12*4], label='nuclear')
# plt.plot(x[0:12], solarProd[len(x)-12*5:len(x)-12*4], label='solar')
# plt.plot(x[0:12], windProd[len(x)-12*5:len(x)-12*4], label='wind')
# plt.plot(x[0:12], hydroProd[len(x)-12*5:len(x)-12*4], label='hydro')
# plt.plot(x[0:12], gasProd[len(x)-12*5:len(x)-12*4], label='gas')
# print(avgActions)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False)

ax1.plot(x[:len(x)-12][len(avgActions)-12*5:], prodPrice[len(avgActions)-12*5:])
ax1.set_title('price of production')
ax1.set_ylim(bottom = min(prodPrice[len(avgActions)-12*5:]), top=max(prodPrice[len(avgActions)-12*5:]))

ax2.plot(x[:len(x)-12][len(avgActions)-12*5:], avgActions[len(avgActions)-12*5:])
ax2.set_title('actions')
ax2.set_ylim(bottom = min(avgActions[len(avgActions)-12*5:]), top=max(avgActions[len(avgActions)-12*5:]))

ax3.plot(x[:len(x)-12][len(avgActions)-12*5:], retailPrice[len(avgActions)-12*5:])
ax3.set_title('price of retail')
ax3.set_ylim(bottom = min(retailPrice[len(avgActions)-12*5:]), top=max(retailPrice[len(avgActions)-12*5:]))

# plt.legend()
f.tight_layout(pad=3.0)
plt.show()

# we need recursive behavior to explore the Q tree, so I'm defining this as a function
# we need to:
	# get the average value of all expected rewards in all leaves of both trees
	# get the average difference between the expected values in all leaves of both trees
# sum1 = 0
# sum2 = 0
# diff = 0
# num = 0
# def calcValues(q1, q2):
# 	global sum1
# 	global sum2
# 	global diff
# 	global num

# 	if(('type' in q1) and (q1['type'] == 'leaf')):
# 		# this block will execute if q is a leaf
# 		num += 1
# 		# iterates accross the actions to change all the alpha values
# 		for i in actionSpace:
# 			sum1 += abs(q1[i]['reward'])
# 			sum2 += abs(q2[i]['reward'])
# 			diff += abs(q1[i]['reward'] - q2[i]['reward'])

# 	else:
# 		# this block will execute if q is a node in the Qtree
# 		for i in q1:
# 			calcValues(q1[i], q2[i])

# newQ = batteries[0].Q

# calcValues(oldQ, newQ)

# print('avg value of 1')
# print(sum1/num)
# print('avg value of 2')
# print(sum2/num)
# print('avg diff')
# print(diff/num)

# # we will now only plot the last n days
# n = 5

# length = n*len(T)

# x = range(0, length)

# prodPriceTilde = prodPrice[(len(prodPrice)-length):]
# retailPriceTilde = retailPrice[(len(retailPrice)-length):]
# avgActionsTilde = avgActions[(len(avgActions)-length):]

# plt.plot(x, prodPriceTilde)
# plt.plot(x, retailPriceTilde)

# plt.show()

# print()
# print('forming results')
# sampleInterval = 50
# toPlot = []
# numIterations = len(gasProd)-sampleInterval
# for i in range(0, numIterations):
# 	sum = 0
# 	for j in range(i,i+sampleInterval):
# 		sum += gasProd[j]
# 	toPlot.append(sum/sampleInterval)

# 	# prints progress, WILL NOT WORK WITH PYTHON2
# 	# erases the last output
# 	print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b', end = '')
# 	# produces new output
# 	print(' %'+str(100*i/numIterations), end = '')

# # adds new line
# print()

# x = range(0, len(toPlot))
# plt.plot(x, toPlot)
# plt.show()

# import json
# obj = {}

# obj['gasProd'] = gasProd

# obj['agentRewards'] = avgActions

# obj['agentActions'] = avgReward

# records = json.dumps(obj)

# f = open('noSocialCostOfCarbonAndRealRetailPricing'+str(numBatteries)+'Agents', 'w')
# f.write(records)
# f.close()