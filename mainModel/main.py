from matplotlib import pyplot as plt
from numpy import arange
from markovSources import SolarPanel, WindTurbine
from transformerBox import TransformerBox
import math
from random import randint

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

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []

# main program loop
# each iteration of this loop represents one day in the model
for dummy in range(0,5):
	hydroSchedule = hydroTarget
	# clears hydroTarget for the next day
	hydroTarget = []

	# each iteration in this loop represents one time interval (not a full day)
	for t in T:
		# resets these variables to zero
		totalProduction = 0
		totalDemand = 0
		priceOfProduction = 0

		# production from nuclear plants is added to the grid, 
		# with necessary adjustments to priceOfProduction
		totalProduction += baseline
		priceOfProduction += nuclearPrice*baseline

		# production from solar panels and wind turbines is added to the grid
		for producer in stochasticProducers:
			renewableProduction = producer.update(t)
			totalProduction += renewableProduction
			priceOfProduction += renewablePrice*renewableProduction

		# users draw electricity from grid
		for i in boxes:
			totalDemand += i.update(t)

		# hydro power matches the difference between totalDemand and power production from all sources except gas
		# and so we must record that difference here
		hydroTarget.append(totalDemand - totalProduction - 300)

		totalProduction += hydroSchedule[t]
		priceOfProduction += hydroPrice*hydroSchedule[t]

		# production from gas-fired plants is added to the grid
		if(totalDemand > totalProduction):
			gasProduction = totalDemand - totalProduction
			totalProduction += gasProduction
			priceOfProduction += gasPrice*gasProduction

		# these lines only serve to make plots below
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(priceOfProduction)

demand = demand[len(T):len(demand)]
production = production[len(T):len(production)]
prodPrice = prodPrice[len(T):len(prodPrice)]

# plots everything all nice and pretty
x = range(0, len(demand))
plt.plot(x, demand)
plt.plot(x, production)
plt.plot(x,prodPrice)
plt.show()
