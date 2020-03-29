from QLearningAgent import QLearningAgent
from matplotlib import pyplot as plt
from random import randint
from copy import copy

class Agent(QLearningAgent):
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

def deepCopy(obj):
	obj = copy(obj)
	if isinstance(obj, dict):
		for i in obj:
			obj[i] = deepCopy(obj[i])
	
	return obj

actionSpace = [0,1]

agents = []
numAgents = 1
for i in range(numAgents):
	# normally we would provide the possible actions available to the agent
	# to the contrustor here, but the default actions of 0 and 1 are good enough.
	agents.append(Agent(actionSpace))

stateCeiling = 1
stateFloor = -1
state = [0,0,0]

# will be used to plot diagnostics 
avgRewards = []

oldQ = {}

numIterations = 500000
for i in range(numIterations):

	# used for diagnostics
	avgReward = 0

	# gets actions and gives rewards from all the agents
	for agent in agents:
		action = agent.getAction(state)
		reward = action*sum(state)

		agent.giveReward(reward)

		# keeps track of the rewards for diagnostics
		avgReward += reward

	# calculates and records the average reward
	avgReward = avgReward/len(agents)
	avgRewards.append(avgReward)

	# randomly sets state
	for j in range(len(state)):
		state[j] = randint(stateFloor, stateCeiling)

	if (i == int(numIterations/2)):
		oldQ = deepCopy(agents[0].Q)
		agents[0].rejuvenate()
		# agents[0].freeze()

# viewSize = 10
# length = len(avgRewards)
# toPlot = []
# for i in range(int(length/viewSize)):
# 	toPlot.append(sum(avgRewards[i*viewSize:(i+1)*viewSize])/viewSize)

# plt.plot(range(len(toPlot)), toPlot)
# plt.show()

# we need recursive behavior to explore the Q tree, so I'm defining this as a function
# we need to:
	# get the average value of all expected rewards in all leaves of both trees
	# get the average difference between the expected values in all leaves of both trees
sum1 = 0
sum2 = 0
diff = 0
num = 0
def calcValues(q1, q2):
	global sum1
	global sum2
	global diff
	global num

	if(('type' in q1) and (q1['type'] == 'leaf')):
		# this block will execute if q is a leaf
		for i in actionSpace:
			sum1 += abs(q1[i]['reward'])
			sum2 += abs(q2[i]['reward'])
			diff += abs(q1[i]['reward'] - q2[i]['reward'])

	else:
		# this block will execute if q is a node in the Qtree
		for i in q1:
			calcValues(q1[i], q2[i])
			num += 1

newQ = agents[0].Q

calcValues(oldQ, newQ)

print('avg value of 1')
print(sum1/num)
print('avg value of 2')
print(sum2/num)
print('avg diff')
print(diff/num)

