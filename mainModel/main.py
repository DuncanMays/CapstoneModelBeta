from matplotlib import pyplot as plt
from numpy import arange
from markovSources import House, Factory, SolarPanel, WindTurbine
import math

# this array holds all the times throughout the day that our model will iterate
# right now I've set it to 15 minute intervals, we should take care in other 
# parts of the program to allow intervals of other sizes
# numpy.arange does the same thing as linspace in Matlab
T = arange(0, 24, 0.25)

# baseline power production, this will be the amount of electricity produced by nuclear
# power plants
baseline = 9779

# I am going to make the hydro production schedule a sin wave just to start out with,
# we should definetely look at IESO data and find a function that fits it better 
hydroSchedule = lambda x : 1900*math.sin(x*math.pi/12) + 1900

# this array will hold all the consumers that draw electricity in a stochastic manner
stochasticConsumers = []
numHouses = 500
numFactories = 20
# creates numHouses instances of the House class
for i in range(0,numHouses):
	stochasticConsumers.append(House())
# creates numFactories instances of the Factory class
for i in range(0,numFactories):
	stochasticConsumers.append(Factory())

# This array holds the producers that produce in a stochastic manner
stochasticProducers = []
numSolarPanels = 50
numWindTurbines = 10
# creates numSolarPanels instances of SolarPanel
for i in range(0,numSolarPanels):
	stochasticProducers.append(SolarPanel())
# creates numWindTurbines instances of WindTurbine
for i in range(0,numWindTurbines):
	stochasticProducers.append(WindTurbine())

# the price of production at a given interval
priceOfProduction = 0

# the price of production for various methods
# made these numbers up, we should run some kind of regression on IESO
# data to get the actual numbers
nuclearPrice = 1
hydroPrice = 2
gasPrice = 3
# I am assuming that it costs the same amount of money to produce electricity
# by both wind and solar, this is probably a bad assumption
renewablePrice = 0.1

# returns the retail price of electricity at the given time, in cents per kWh
# lambda functions don't like if statements, so I had to use an actual function
# this is Ontario's winter pricing scheme
def priceOfConsumption(time):
	if(time < 7):
		return 6.5
	elif(time < 11):
		return 13.4
	elif(time < 17):
		return 9.4
	elif(time < 19):
		return 13.4
	else:
		return 6.5

# arrays that will be used to plot data at the end of the program, serve no other purpose than this
demand = []
production = []
prodPrice = []

# main program loop
# each iteration of this loop represents one day in the model
# while(True):
# for testing purposes, I've commented out the origional while statement and replaced it with a for loop
# this is so that the loop will only iterate once, which makes testing possible
for dummy in range(0,1):
	# each iteration in this loop represents one time interval (not a full day)
	for t in T:
		# resets these variables to zero
		totalProduction = 0
		totalDemand = 0
		priceOfProduction = 0

		# production from nuclear and hydro plants is added to the grid, 
		# with necessary adjustments to priceOfProduction
		totalProduction += baseline
		priceOfProduction += nuclearPrice*baseline
		totalProduction += hydroSchedule(t)
		priceOfProduction += hydroPrice*hydroSchedule(t)

		# production from solar panels and wind turbines is added to the grid
		for producer in stochasticProducers:
			renewableProduction = producer.update(t)
			totalProduction += renewableProduction
			priceOfProduction += renewablePrice*renewableProduction

		# users draw electricity from grid
		for consumer in stochasticConsumers:
			totalDemand += consumer.update(t)

		# production from gas-fired plants is added to the grid
		if(totalDemand > totalProduction):
			gasProduction = totalDemand - totalProduction
			totalProduction += gasProduction
			priceOfProduction += gasPrice*gasProduction

		# these lines only serve to make plots below
		demand.append(totalDemand)
		production.append(totalProduction)
		prodPrice.append(priceOfProduction)

# plots everything all nice and pretty
plt.plot(T, demand)
plt.plot(T, production)
plt.plot(T,prodPrice)
plt.show()
