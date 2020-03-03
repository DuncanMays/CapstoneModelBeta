print('importing required modules')

from matplotlib import pyplot as plt
from numpy import arange
from markovSources import SolarPanel, WindTurbine
from transformerBox import TransformerBox
import math
from random import randint, choice
from QLearningAgent import QLearningAgent

print('setting up')

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
		self. transformerBox = transformerBox
		
		super(Battery, self).__init__(actions)

	def getAction(self, state):
		# quantizing local demand to an integer within 1 and 10, i have no idea if this is a good quantization
		demand = quantize(self.transformerBox.totalDemand, arange(0, 10, 1))

		# if we keep actions to sets that shares a factor we can avoid discretizing charge

		# adds demand and charge to state so that the underlying Q-learning agent can 'see' them
		state.append(demand)
		state.append(self.charge)

		action = super(Battery, self).getAction(state)

		# we now must check to make sure that the action makes sense
		# if we sell, have we sold more than the battery's charge?
		# if we buy, will we reach the battery's capacity?

		# negative action signifies selling electricity
		if (action < 0) and (-action > self.charge):
			# if we sell more charge than we have
			action = -self.charge
			self.charge += action
		# I have assumed that the battery's max capacity is 10
		elif (action + self.charge > 10):
			# if we buy more charge than we have capacity for
			action = 10 - self.charge

		return action

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes, as we may want to change 
# this as our Q-learning algorithm evolves.
# numpy.arange does the same thing as linspace in Matlab
T = arange(0, 24, 1)

# baseline power production, this will be the amount of electricity produced by nuclear
# power plants
baseline = 3500

# This array holds the producers that produce in a stochastic manner
stochasticProducers = []
numSolarPanels = 200
numWindTurbines = 50
# creates numSolarPanels instances of SolarPanel
for i in range(0,numSolarPanels):
	stochasticProducers.append(SolarPanel())
# creates numWindTurbines instances of WindTurbine
for i in range(0,numWindTurbines):
	stochasticProducers.append(WindTurbine())

# doesn't really matter what hydroTarget is initialized it, it will be updated to
#  something usefull on the first cyle of the model. So long as whatever it is has the 
#  same number of elements as T, the model will work. For this reason i've initialized
#  it to T
hydroTarget = T

# the price of production at a given interval, initialized to zero
priceOfProduction = 0

# the price of production for various methods
# made these numbers up, we should run some kind of regression on IESO
# data to get the actual numbers
nuclearPrice = 0.5
hydroPrice = 1
gasPrice = 1.5
# I am assuming that it costs the same amount of money to produce electricity
# by both wind and solar, this is probably a bad assumption
renewablePrice = 0.5

# creates the power boxes through which demand will flow and a Qlearning agent will take action
numBoxes = 100
boxes = []
for i in range(0, numBoxes):
	# the first parameter is the number of total demand agents the box will service
	# the second parameter is the ratio between the number of houses and the number of factories served.
	boxes.append(TransformerBox(randint(20, 50), 0.9))

# we will now attach batteries to some of the boxes
numBatteries = 50
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
		battery = Battery([-1,0,1], box)
		batteries.append(battery)
for i in range(0, numBatteries):
	# assigns a batter to a random box
	assignBattery(choice(boxes))

# this is something that needs to change to a function that at least somewhat mimics reality
priceOfRetail = 5

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []

print('starting model')

# main program loop
# each iteration of this loop represents one day in the model
for day in range(0, 5):
	print('day: '+str(day))

	# hydro power will try to match the power defecit of the day before, so while
	#  hydroSchedule is read from in the loop below, hydroTarget will be written to
	hydroSchedule = hydroTarget

	# clears hydroTarget for the next day
	hydroTarget = []

	# each iteration in this loop represents one time interval (not a full day)
	for t in T:
		# resets these variables to zero
		totalProduction = 0
		totalDemand = 0
		priceOfProduction = 0

		# production  from uncontrolled sources must be added to the grid first, 
		#  as all other variables (demand, production from controlled sources like 
		#  hydro and gas, as well as discharge from batteries) are determined using it.

		# production from nuclear plants is added to the grid, 
		# with necessary adjustments to priceOfProduction
		totalProduction += baseline
		priceOfProduction += nuclearPrice*baseline

		# production from solar panels and wind turbines is added to the grid
		for producer in stochasticProducers:
			renewableProduction = producer.update(t)
			totalProduction += renewableProduction
			priceOfProduction += renewablePrice*renewableProduction

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
		hydroTarget.append(totalDemand - totalProduction - 300)

		# adds hydro power to the grid
		totalProduction += hydroSchedule[t]
		priceOfProduction += hydroPrice*hydroSchedule[t]

		# production from gas-fired plants is added to the grid
		if(totalDemand > totalProduction):
			gasProduction = totalDemand - totalProduction
			totalProduction += gasProduction
			priceOfProduction += gasPrice*gasProduction

		# we do Qlearning last. I had a bit of a dillemma on the order in which to implement this. Qlearning agents
		#  need to know the price of production, which is determined by global demand, which is determined by the 
		#  actions of Qlearning agents. What I've decided to do is push supply and demand from Qlearning agents into
		#  the next timestep. This means that whatever electricity an agent buys or sells will be drawn from or 
		#  pushed to the grid in the next timestep. These values are stored in the variables Qdemand and Qsupply.
		Qdemand = 0
		Qsupply = 0
		actions = []

		# get agents actions from time, price of retail, price of production
		# the local demand, as well as the battery's capacity, will be added to state within the battery class.
		for j in batteries:
			actions.append(j.getAction([t, priceOfRetail, quantize(priceOfProduction, arange(35000, 40000, 500))]))

		# calculate reward the reward for each agent
		for j in range(0, len(batteries)):
			action = actions[j]
			if action < 0:
				# the agent sold
				batteries[j].giveReward(-priceOfRetail*action)
			else:
				# the agent bought
				batteries[j].giveReward(priceOfProduction*action)

		# these lines only serve to make plots below
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(priceOfProduction)

# the first day is tainted data, as hydroSchedule is not set to anything useful, we will cut it out of the data
demand = demand[len(T):len(demand)]
production = production[len(T):len(production)]
prodPrice = prodPrice[len(T):len(prodPrice)]

# plots everything all nice and pretty
x = range(0, len(demand))
plt.plot(x, demand)
plt.plot(x, production)
plt.plot(x,prodPrice)
plt.show()
