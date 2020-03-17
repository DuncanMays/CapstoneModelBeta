print('importing required modules')

from matplotlib import pyplot as plt
from numpy import arange
from markovSources import House, SolarPanel, WindTurbine
from transformerBox import TransformerBox
import math
from random import randint, choice
from QLearningAgent import QLearningAgent

print('setting up')

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes, as we may want to change 
# this as our Q-learning algorithm evolves.
# numpy.arange does the same thing as linspace in Matlab
T = arange(0, 24, 4)

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

numChargeCells = 3
chargeSpace = []

actionSpace = [-1,0,1]

# this is the first example of a pattern that repeats itself a bunch of times,
# we determine a maximum vlaue a quantity can take, and a minimum quatnity. and then we form a 
# quantization with a given number of cells in that range. In this case we are quantizint the
# battery's charge. This needs to happen before we define the batter class below.
minCharge = 0
maxCharge = 15

interval = (maxCharge - minCharge)/numChargeCells
chargeSpace = arange(minCharge, maxCharge, interval)

# this method takes a value and returns the closest element of the set it is given
def quantize(value, targetSet):
	closestElement = targetSet[0]
	for i in targetSet:
		if (abs(value - i) < abs(value - closestElement)):
			closestElement = i

	return closestElement

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

	def chooseAction(self, priceOfProduction, priceOfRetail, time):
		# quantizing local demand, pulling totalDemand from the box will only work if get action
		# is called after update is called on the box.
		localDemand = quantize(self.transformerBox.totalDemand, self.transformerBox.demandSpace)

		charge = quantize(self.charge, chargeSpace)

		prodPrice = quantize(priceOfProduction, prodPriceSpace)

		retailPrice = quantize(priceOfRetail, retailPriceSpace)

		time = quantize(time, T)

		state = [time, prodPrice, retailPrice, localDemand, charge]

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
		reward = reward*10

		self.rewards.append(reward)
		self.profit += reward

		super(Battery, self).giveReward(reward)


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
numSolarPanels = 20
# creates numSolarPanels instances of SolarPanel
for i in range(0,numSolarPanels):
	solarPanels.append(SolarPanel())
	prodPriceMin += solarPrice*solarPanels[i].min
	prodPriceMax += solarPrice*solarPanels[i].max

WindTurbines = []
numWindTurbines = 50
# creates numWindTurbines instances of WindTurbine
for i in range(0,numWindTurbines):
	WindTurbines.append(WindTurbine())
	prodPriceMin += windPrice*WindTurbines[i].min
	prodPriceMax += windPrice*WindTurbines[i].max

# prodPrice min and max will take hydro power into account when the hydro schedule is set below in the loop

# creates the power boxes through which demand will flow and a Qlearning agent will take action
numBoxes = 500
boxes = []
for i in range(0, numBoxes):
	# the first parameter is the number of total demand agents the box will service
	# the second parameter is the ratio between the number of houses and the number of factories served.
	boxes.append(TransformerBox(randint(20, 50), 0.9, numCells = localDemandCells))

# we will now attach batteries to some of the boxes
numBatteries = 300
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

# we then quantize this range by the number of demand cells we want
interval = (minGlobalDemand - maxGlobalDemand)/globalDemandCells
globalDemandSpace = arange(minGlobalDemand, maxGlobalDemand, interval)

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
prodPriceSpace = arange(prodPriceMin, prodPriceMax, interval)

# this is something that needs to change to a function that at least somewhat mimics reality
def retailprice(time):
    if 0<= time < 7:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0015
    elif 7 <= time < 11:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0025
    elif 11 <= time < 17:
        #On peak rates, 13.4¢/kWh, 13.4¢ / 100(¢/$) * 1000 kWh/MWh = 134$/MWh
        # priceOfRetail = 134
        priceOfRetail = 0.0030
    elif 17 <= time < 19:
        #Mid peak rates, 9.4¢/kWh, 9.4¢ / 100(¢/$) * 1000 kWh/MWh = 94$/MWh
        # priceOfRetail = 94
        priceOfRetail = 0.0025
    elif 19 <= time <= 24:
        #Off peak rates, 6.5¢/kWh, 6.5¢ / 100(¢/$) * 1000 kWh/MWh = 65$/MWh
        # priceOfRetail = 65
        priceOfRetail = 0.0015
    
    return priceOfRetail

retailPriceSpace = [65, 94, 134]


# the price of production at a given interval, initialized to zero
priceOfProduction = 0

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []
retailPrice = []
gasProd = []

print('starting model')

# main program loop
# each iteration of this loop represents one day in the model
for day in range(0, 5):
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
		priceOfProduction = 0
		priceOfRetail = retailprice(t)

		# production  from uncontrolled sources must be added to the grid first, 
		#  as all other variables (demand, production from controlled sources like 
		#  hydro and gas, as well as discharge from batteries) are determined using it.

		# production from nuclear plants is added to the grid, 
		# with necessary adjustments to priceOfProduction
		totalProduction += baseline
		priceOfProduction += nuclearPrice*baseline

		# production from solar panels is added to the grid
		for panel in solarPanels:
			temp = panel.update(t)
			totalProduction += temp
			priceOfProduction += solarPrice*temp

		# production from wind turbines is also added to the grid
		for turbine in WindTurbines:
			temp = turbine.update(t)
			totalProduction += temp
			priceOfProduction += windPrice*temp

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
		diff = totalDemand - totalProduction
		if (diff < 0):
			# checks to make sure hydro production cant be negative
			diff = 0
		hydroTarget[t] = diff

		# adds hydro power to the grid
		totalProduction += hydroSchedule[t]
		priceOfProduction += hydroPrice*hydroSchedule[t]

		# production from gas-fired plants is added to the grid
		gasProduction = 0
		if(totalDemand > totalProduction):
			gasProduction = totalDemand - totalProduction
			totalProduction += gasProduction
			priceOfProduction += gasPrice*gasProduction
		gasProd.append(gasProduction)

		# we do Qlearning last. I had a bit of a dillemma on the order in which to implement this. Qlearning agents
		#  need to know the price of production, which is determined by global demand, which is determined by the 
		#  actions of Qlearning agents. What I've decided to do is push supply and demand from Qlearning agents into
		#  the next timestep. This means that whatever electricity an agent buys or sells will be drawn from or 
		#  pushed to the grid in the next timestep. These values are stored in the variables Qdemand and Qsupply.
		Qdemand = 0
		Qsupply = 0
		actions = []

		# right now, price of production is the total price to produce all the electricity in the system, so we must divide
		# it by the amount of electricity in the system to get the price per MWh
		priceOfProduction = priceOfProduction/totalProduction 

		if (day != 0):
			# get agents actions from time, price of retail, price of production
			# the local demand, as well as the battery's capacity, will be added to state within the battery class.
			for j in batteries:
				action = j.chooseAction(priceOfProduction, priceOfRetail, t)
				actions.append(action)

				if action > 0:
					# if the agent bought
					totalDemand += action
				elif action < 0:
					# if the agent sold
					totalProduction -= action

			# calculate reward the reward for each agent
			for j in range(0, len(batteries)):
				action = actions[j]
				if action < 0:
					# the agent sold
					batteries[j].giveReward(priceOfRetail*action)
				else:
					# the agent bought
					batteries[j].giveReward(priceOfProduction*action)

		# these lines only serve to make plots below
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(priceOfProduction)
		retailPrice.append(priceOfRetail)

	if(day == 0):

		# finds the max and min hydro production values
		max = hydroTarget[0]
		min = hydroTarget[0]
		for i in hydroTarget:
			if (hydroTarget[i] > max):
				max = hydroTarget[i]

			elif (hydroTarget[i] < min):
				min = hydroTarget[i]

		maxGlobalProd += max
		minGlobalProd += min

		prodPriceMax += hydroPrice*max
		prodPriceMin += hydroPrice*min

		prodPriceMax = prodPriceMax/maxGlobalProd
		prodPriceMin = prodPriceMin/minGlobalProd

		interval = (prodPriceMax - prodPriceMin)/prodPriceCells
		prodPriceSpace = arange(prodPriceMin, prodPriceMax, interval)

# the first day is tainted data, as hydroSchedule is not set to anything useful, we will cut it out of the data
demand = demand[len(T):len(demand)]
production = production[len(T):len(production)]
prodPrice = prodPrice[len(T):len(prodPrice)]
retailPrice = retailPrice[len(T):len(retailPrice)]

# plots everything all nice and pretty
x = range(0, len(demand))
# plt.plot(x, demand)
# plt.plot(x, [maxGlobalDemand for i in range(0, len(x))])
# plt.plot(x, [minGlobalDemand for i in range(0, len(x))])
# plt.plot(x, production)
# plt.plot(x, [maxGlobalProd for i in range(0, len(x))])
# plt.plot(x, [minGlobalProd for i in range(0, len(x))])
# plt.plot(x, prodPrice)
# plt.plot(x, batteries[0].toPlot)
# plt.plot(x, [prodPriceMax for i in range(0, len(x))])
# plt.plot(x, [prodPriceMin for i in range(0, len(x))])
# plt.plot(x, retailPrice)
# plt.plot(range(0, len(batteries[0].rewards)), batteries[0].rewards)
plt.show()

# for i in batteries:
# 	print(i.profit)


