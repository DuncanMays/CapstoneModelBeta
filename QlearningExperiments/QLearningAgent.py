# Written by Duncan Mays in January 2020 for his fourth year Capstone Project

from random import uniform, choice

'''
This class will use Q learning to maximize the reward it recieves

There are 2 important functions, getAction and giveReward.

getAction takes a list representing the state, and uses it to look up the appropriate policy in the Q-table,
 from which it returns the action, either by random selection or the one with the highest expected reward.

giveReward takes a float and adjusts the last policy to reflect the reward it has recieved.

It is very important to call getAction and giveReward in alternating order, otherwise the agent will not learn properly.

I have taken an unconventional approach to encoding the Q-table. Instead of using a 2D list (table), I have used nested
 dicts in a sort of tree configuration. The advantage of this is that we do not have to 'set up' the Q-table on
 instantiating the class. The Q-tree will grow dynamically as new states are discovered, and until all states have been
 discovered. It also means that we don't need to translate to indices from states and vice versa, the states themselves
 can be used as keys to access policy, be they integers, floats or strings. See getPolicy() for how the tree is
 traversed and newPolicy() for how the tree grows upon discovery of a new state.

Policy will be encoded as a dict as well, the keys will be each possible action and the values they lead to will be the
 expected reward upon taking that action. So each policy, each 'leaf' on the Q-tree, will look like this:
 {action1: e1, action2, e2, ........} where ei represents the expected reward upon taking action i. See newPolicy().

I have also switched the time-direction of updating expected rewards. Normally, when updating the Q-table, the agent will
 'look ahead' in the next state to see if its a poor position. Due to the stochastic nature of the system, it is
 difficult to predict the next state, even knowing the action taken in the current state. As a result, instead this agent
 will 'look back', it will remember the previouse state and action, and adjust the expected reward of that combination to 
 reflect the current reward. This will allow the agent to know if an action will lead to a poor position. See giveReward()
 for how the policies are updated.
'''
class QLearningAgent:
	
	'''
	actions is an array that gives the possible actions the agent can take
	learningRate controls how fast the agent learns, the higher it is the faster
	discount controls how much the agent values future rewards, high discount values mean the agent 'plans ahead' more
	exploration controls how likely the agent is to randomly select policy
	'''
	def __init__(self, actions, learningRate=0.1 , discount=0.25 , exploration=0.1):

		self.Q = {}
		self.actions = actions
		self.learningRate = learningRate
		self.discount = discount
		self.exploration = exploration

		# these variable will store the state/action combination at the previous time-step
		self.secondLastState = 'bootstrap'
		self.secondLastAction = 'bootstrap'

		# stores the state/action combination in the current time-step
		self.lastState = 'bootstrap'
		self.lastAction = 'bootstrap'

		# because the agent remembers the previous state/action combination, it will glitch
		#  on the first call to giveReward(), since in that case there was no previous 
		#  state/action combination. This call to getAction() removes that problem.
		self.getAction(['bootstrap'])

	def newPolicy(self, state):
		if (len(state) > 0):
			return {state[0]: self.newPolicy(state[1:len(state)])}
		else:
			q = {}
			for i in self.actions:
				q[i] = uniform(-1, 1)
			return q

	# recursively traverses the Q-tree
	def getPolicy(self, state, q):
		if(len(state) > 0):
			try:
				return self.getPolicy(state[1:len(state)], q[state[0]])
			except(KeyError):
				q[state[0]] = self.newPolicy(state[1:len(state)])
				return self.getPolicy(state[1:len(state)], q[state[0]])
		else:
			return q

	def getAction(self, state):
		if (uniform(0, 1) < self.exploration):
			action = choice(self.actions)
		else:
			q = self.getPolicy(state, self.Q)

			action = self.actions[0]
			for i in self.actions:
				if (q[i] > q[action]):
					action = i

		self.secondLastState = self.lastState
		self.secondLastAction = self.lastAction

		self.lastState = state
		self.lastAction = action

		return action

	def giveReward(self, reward):
		policy = self.getPolicy(self.lastState, self.Q)

		alpha = self.learningRate
		beta = 1 - alpha

		temp = beta*policy[self.lastAction] + alpha*reward

		policy[self.lastAction] = temp

		policy = self.getPolicy(self.secondLastState, self.Q)

		policy[self.secondLastAction] += self.learningRate*self.discount*temp 



