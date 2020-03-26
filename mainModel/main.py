print('importing required modules')

from matplotlib import pyplot as plt
from numpy import arange
import numpy as np
from markovSources import House, SolarPanel, WindTurbine
from transformerBox import TransformerBox
import math
from random import randint, choice, shuffle
from QLearningAgent import QLearningAgent
from copy import copy

print('setting up')

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes, as we may want to change 
# this as our Q-learning algorithm evolves.
# numpy.arange does the same thing as linspace in Matlab
T = np.arange(0, 24, 2)

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
prodPriceCells = 5
prodPriceSpace = []

numChargeCells = 5
chargeSpace = []

actionSpace = [-500,0,500]

# this is the first example of a pattern that repeats itself a bunch of times,
# we determine a maximum vlaue a quantity can take, and a minimum quatnity. and then we form a 
# quantization with a given number of cells in that range. In this case we are quantizint the
# battery's charge. This needs to happen before we define the batter class below.
minCharge = 0
maxCharge = 15000

interval = (maxCharge - minCharge)/numChargeCells
chargeSpace = arange(minCharge, maxCharge, interval)

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

	def __init__(self, actions, transformerBox):
		self.charge = 0
		self.transformerBox = transformerBox

		# this array will be used to track the rewards of the agent, for performance evaluation
		self.rewards = []
		self.profit = 0

		self.toPlot = []

		# calling the init method of the parent class
		super(Battery, self).__init__(actions)

	def chooseAction(self, totalCostProduction, priceOfRetail, time):
		# quantizing local demand, pulling totalDemand from the box will only work if get action
		# is called after update is called on the box.
		# localDemand = quantize(self.transformerBox.totalDemand, self.transformerBox.demandSpace)

		charge = quantize(self.charge, chargeSpace)

		prodPrice = quantize(totalCostProduction, prodPriceSpace)

		# retailPrice = quantize(priceOfRetail, retailPriceSpace)

		time = quantize(time, T)

		state = [time, prodPrice, charge]

		action = super(Battery, self).getAction(state)

		self.toPlot.append(prodPrice)

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

	def giveReward(self, reward):

		# the proffits are so tiny I worry they won't change the Q table enough
		reward = 10*reward

		self.rewards.append(reward)
		self.profit += reward

		super(Battery, self).giveReward(reward)

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

# represents gas furnaces, not a lot of complexity, only needed as a way 
#  to distinguish from things like Qlearning agents or demand sources.
class GasFurnace():
	def __init__(self, maxOutput):
		self.maxOutput = maxOutput

# doesn't really matter what hydroTarget is initialized it, it will be updated to
#  something usefull on the first cyle of the model. So long as whatever it is has the 
#  same number of elements as T, the model will work. For this reason i've initialized
#  it to T
hydroTarget = {T[i]: 0 for i in range(0, len(T))}

# the price of production for various methods in dollars per dollars / (megawatt/Hours)^2
nuclearPrice = 0.00032
hydroPrice = 0.00790
gasPrice = 0.00975
windPrice = -0.00130
solarPrice = -0.02063

# we will need these to create a quantization for the production price of electricity
prodPriceMin = 0
prodPriceMax = 0

solarPanels = []
numSolarPanels = 200
# creates numSolarPanels instances of SolarPanel
for i in range(0,numSolarPanels):
	solarPanels.append(SolarPanel())
	prodPriceMin += solarPrice*solarPanels[i].min
	prodPriceMax += solarPrice*solarPanels[i].max

WindTurbines = []
numWindTurbines = 30
# creates numWindTurbines instances of WindTurbine
for i in range(0,numWindTurbines):
	WindTurbines.append(WindTurbine())
	prodPriceMin += windPrice*WindTurbines[i].min
	prodPriceMax += windPrice*WindTurbines[i].max

# prodPrice min and max will take hydro power into account when the hydro schedule is set below in the loop

# creates the power boxes through which demand will flow and a Qlearning agent will take action
numBoxes = 650
boxes = []
for i in range(0, numBoxes):
	# the first parameter is the number of total demand agents the box will service
	# the second parameter is the ratio between the number of houses and the number of factories served.
	boxes.append(TransformerBox(randint(20, 50), 0.9, numCells = localDemandCells))

# shuffles them into a random order
shuffle(boxes)

# we will now attach batteries to some of the boxes, not that the boxes are in random order so the 
#  batteries will be assigned randomly
numBatteries = 200

if(numBatteries > len(boxes)):
	print('we cant have more batteries than boxes')

batteries = []
for i in range(0, numBatteries):
	boxes[i].containsAgent = True
	batteries.append(Battery(actionSpace, boxes[i]))


# we now create the space in which global demand will live in, we will not
# use it right now but we very well might when we implement post-procing.
# we do this by first determining the range that demand can take values in
minGlobalDemand = sum([boxes[i].minLocalDemand for i in range(0, len(boxes))])
maxGlobalDemand = sum([boxes[i].maxLocalDemand for i in range(0, len(boxes))])

# we then quantize this range by the number of demand cells we want
interval = (minGlobalDemand - maxGlobalDemand)/globalDemandCells
globalDemandSpace = np.arange(minGlobalDemand, maxGlobalDemand, interval)

# baseline power production, this will be the amount of electricity produced by nuclear
# power plants
baseline = minGlobalDemand

prodPriceMin += nuclearPrice*baseline
prodPriceMax += nuclearPrice*baseline

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

prodPriceMin += hydroPrice*minHydro
prodPriceMax += hydroPrice*maxHydro

# we must normalize
prodPriceMax = prodPriceMax/maxGlobalProd
prodPriceMin = prodPriceMin/minGlobalProd

interval = (prodPriceMax - prodPriceMin)/prodPriceCells
prodPriceSpace = np.arange(prodPriceMin, prodPriceMax, interval)

# this is something that needs to change to a function that at least somewhat mimics reality
def retailprice(time):
    if 0<= time < 7:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0005
    elif 7 <= time < 11:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0015
    elif 11 <= time < 17:
        #On peak rates, 13.4¢/kWh, 13.4¢ / 100(¢/$) * 1000 kWh/MWh = 134$/MWh
        # priceOfRetail = 134
        priceOfRetail = 0.0025
    elif 17 <= time < 19:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0015
    elif 19 <= time <= 24:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0005
    
    return priceOfRetail

retailPriceSpace = [65, 94, 134]

# we now will create an array of gas furnaces
numFurnaces = 10
maxOutput = 20000
furnaces = []
for i in range(0, numFurnaces):
	furnaces.append(GasFurnace(maxOutput))

# the price of production at a given interval, initialized to zero
totalCostProduction = 0

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []
retailPrice = []
gasProd = []
avgActions = []
solarProduction = []
windProduction = []

print('starting model')

# main program loop
# each iteration of this loop represents one day in the model
for day in range(0, 500):
	print('day: '+str(day))

	# hydro power will try to match the power defecit of the day before, so while
	#  hydroSchedule is read from in the loop below, hydroTarget will be written to
	hydroSchedule = copy(hydroTarget)

	# clears hydroTarget for the next day
	hydroTarget = {}

	# each iteration in this loop represents one time interval (not a full day)
	for t in T:
		# resets these variables to zero
		totalProduction = 0
		totalDemand = 0
		totalCostProduction = 0
		priceOfRetail = retailprice(t)

		# production  from uncontrolled sources must be added to the grid first, 
		#  as all other variables (demand, production from controlled sources like 
		#  hydro and gas, as well as discharge from batteries) are determined using it.

		# production from nuclear plants is added to the grid, 
		# with necessary adjustments to totalCostProduction
		totalProduction += baseline
		totalCostProduction += nuclearPrice*baseline

		# production from solar panels is added to the grid
		total = 0
		for panel in solarPanels:
			temp = panel.update(t)
			totalProduction += temp
			totalCostProduction += solarPrice*temp
			total += temp

		solarProduction.append(total)

		# production from wind turbines is also added to the grid
		total = 0
		for turbine in WindTurbines:
			temp = turbine.update(t)
			totalProduction += temp
			totalCostProduction += windPrice*temp
			total += temp

		windProduction.append(total)

		# consumption must happen before any Q learning agents take action, and
		#  before hydro and gas, as this will determine the electricity defecit,
		#  which hydro will make up for in the next day and gas will make up for
		#  today is one of the parameters of the learning agents. This might get
		#  tricky if we ever implement post-pricing, as demand will be determined
		#  by price.

		#  users draw electricity from grid
		for i in boxes:
			totalDemand += i.update(t)

		# hydro power matches the difference between totalDemand and nuclear power and stochastic producers
		# and so we must record that difference here
		diff = totalDemand - totalProduction - 1000
		if (diff < 0):
			# checks to make sure hydro production cant be negative
			diff = 0
		hydroTarget[t] = diff

		# adds hydro power to the grid
		totalProduction += hydroSchedule[t]
		totalCostProduction += hydroPrice*hydroSchedule[t]

		actionItems = []

		actionItems += furnaces
		actionItems += batteries

		shuffle(actionItems)

		gasSum = 0
		actionSum = 0
		for source in actionItems:

			# the first day is a calibration day, so we musn't produce any gas power or do any q learning on the first day
			if(day == 0):
				break

			if isinstance(source, Battery):
				priceOfProduction = totalCostProduction/totalProduction

				action = source.chooseAction(priceOfProduction, priceOfRetail, t)

				reward = 0
				if action > 0:
					# if the agent bought
					totalDemand += action
					reward = -priceOfProduction*action
				elif action < 0:
					# if the agent sold
					totalProduction -= action
					# reward = -priceOfRetail*action
					reward = -priceOfProduction*action

				source.giveReward(reward)
				actionSum += action

			elif isinstance(source, GasFurnace):
				gasProduction = min(source.maxOutput, totalDemand - totalProduction)
				gasProduction = max(gasProduction, 0)
				totalProduction += gasProduction
				totalCostProduction += gasPrice*gasProduction
				gasSum += gasProduction
			else:
				print('this should not happen')
				print(source)

		gasProd.append(gasSum)
		avgActions.append(actionSum)

		# these lines only serve to make plots below
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(totalCostProduction)
		retailPrice.append(priceOfRetail)

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

		prodPriceMax += hydroPrice*maxHydro
		prodPriceMin += hydroPrice*minHydro

		prodPriceMax = prodPriceMax/maxGlobalProd
		prodPriceMin = prodPriceMin/minGlobalProd

		# sets up the quantization for prodPrice
		interval = (prodPriceMax - prodPriceMin)/prodPriceCells
		prodPriceSpace = np.arange(prodPriceMin, prodPriceMax, interval)

	# if (day == 3000):
	# 	print('freezing all agents but one')

	# 	batteries[0].rejuvenate()

	# 	oldQ = deepCopy(batteries[0].Q)

	# 	for i in range(1, len(batteries)):
	# 		batteries[i].freeze()

# a 2D array, the first dimension specifying the agent, the second the timestep
# we dont need to cut out the first day as we do with other stats since q learning isn't active on the first day
rewards = [agent.rewards for agent in batteries]
avgRewards = [sum([rewards[i][j] for i in range(len(batteries))])/len(batteries) for j in range(0, len(rewards[0]))]

# the first day is tainted data as it is a callibration day, we will cut it out of the data
demand = demand[len(T):len(demand)]
production = production[len(T):len(production)]
prodPrice = prodPrice[len(T):len(prodPrice)]
retailPrice = retailPrice[len(T):len(retailPrice)]
solarProduction = solarProduction[len(T):len(solarProduction)]
windProduction = windProduction[len(T):len(windProduction)]
gasProd = gasProd[len(T):len(gasProd)]
avgActions = avgActions[len(T):len(avgActions)]

# it should not be this complicated to serialize numpy arrays, not happy
import io, json
def serializeNumpyArray(array):
	memFile = io.BytesIO()
	np.save(memFile, array)
	memFile.seek(0)
	return json.dumps(memFile.read().decode('latin-1'))

def deserializeNumpyArray(string):
	memFile = io.BytesIO()
	memFile.write(json.loads(string).encode('latin-1'))
	memFile.seek(0)
	return np.load(memFile, allow_pickle=True)

toSave = {}
toSave['avgRewards'] = serializeNumpyArray(avgRewards)
toSave['demand'] = serializeNumpyArray(demand)
toSave['production'] = serializeNumpyArray(production)
toSave['prodPrice'] = serializeNumpyArray(prodPrice)
toSave['retailPrice'] = serializeNumpyArray(retailPrice)
toSave['solarProduction'] = serializeNumpyArray(solarProduction)
toSave['windProduction'] = serializeNumpyArray(windProduction)
toSave['gasProd'] = serializeNumpyArray(gasProd)
toSave['avgActions'] = avgActions

toSaveStr = json.dumps(toSave)

f = open('modelResults', 'w')
f.write(toSaveStr)
f.close()

# numDaysInView = 5
# demand = demand[(len(demand)-numDaysInView*len(T)):len(demand)]
# production = production[(len(production)-numDaysInView*len(T)):len(production)]
# prodPrice = prodPrice[(len(prodPrice)-numDaysInView*len(T)):len(prodPrice)]
# retailPrice = retailPrice[(len(retailPrice)-numDaysInView*len(T)):len(retailPrice)]
# solarProduction = solarProduction[(len(solarProduction)-numDaysInView*len(T)):len(solarProduction)]
# windProduction = windProduction[(len(windProduction)-numDaysInView*len(T)):len(windProduction)]
# gasProd = gasProd[(len(gasProd)-numDaysInView*len(T)):len(gasProd)]
# avgActions = avgActions[(len(avgActions)-numDaysInView*len(T)):len(avgActions)]

# plots everything all nice and pretty
# x = range(0, len(demand))
# plt.plot(x, demand, label="demand")
# plt.plot(x, [maxGlobalDemand for i in range(0, len(x))])
# plt.plot(x, [minGlobalDemand for i in range(0, len(x))])
# plt.plot(x, production, label="supply")
# plt.plot(x, [maxGlobalProd for i in range(0, len(x))])
# plt.plot(x, [minGlobalProd for i in range(0, len(x))])
# plt.plot(x, prodPrice)
# plt.plot(x, [prodPriceMax for i in range(0, len(x))])
# plt.plot(x, [prodPriceMin for i in range(0, len(x))])
# plt.plot(x, retailPrice)
# plt.plot(x, batteries[0].toPlot)
# plt.plot(x, avgActions)
# plt.plot(x, avgActions)
# plt.plot(x, solarProduction, label="solar")
# plt.plot(x, windProduction, label="wind")
# plt.plot(x, gasProd, label="gas")
# plt.plot(x, avgActions, label="agent's output")

# plt.title('excluding priceOfRetail')
# plt.legend()
# plt.show()

# # we need recursive behavior to explore the Q tree, so I'm defining this as a function
# we need to:
# 	# get the average value of all expected rewards in all leaves of both trees
# 	# get the average difference between the expected values in all leaves of both trees
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
# 		# iterates accross the actions to change all the alpha values
# 		for i in actionSpace:
# 			sum1 += abs(q1[i]['reward'])
# 			sum2 += abs(q2[i]['reward'])
# 			diff += abs(q1[i]['reward'] - q2[i]['reward'])

# 	else:
# 		# this block will execute if q is a node in the Qtree
# 		for i in q1:
# 			calcValues(q1[i], q2[i])
# 			num += 1

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