from matplotlib import pyplot as plt
from numpy import arange
from numpy.random import normal
import math

# stochastic experimentation

def chooseAction(diff):
	return normal(diff/3, 0.5)

# an array from 0 to 24 with interval 0.25 (15 minutes)
t = arange(0, 24, 0.25)

schedule  = []
for i in t:
	schedule.append(math.sin(i*math.pi/12))

state = 10
states = []
for i in range(0, len(t)):
	states.append(state)
	state -= chooseAction(state - schedule[i])

plt.plot(t,schedule)
plt.plot(t, states)
plt.show()