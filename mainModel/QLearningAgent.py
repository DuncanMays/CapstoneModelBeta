# Written by Duncan Mays in January 2020 for his fourth year Capstone Project

from random import uniform, choice
import yaml

'''
This class will use Q learning to maximize the reward it recieves

There are 2 important functions, getAction and giveReward, and one important datastructure, which I call leaf.

Each state that the agent comes across will hace a leaf assigned to it. Leafs are dicts with the actions availables to
 the agents as keys. These keys will point to a secondary dict with 2 keys, 'reward' and 'alpha'. As the names imply, 
 reward gives the expected reward of taking that action in that state, and alpha represents the alpha factor that is
 used in the Qlearnign algorithm, it decreases with time to ensure convergence. More specifically, each state's alpha
 value will be the inverse of the number of times that state has been visited. See giveREward for how this is done.

getAction takes a list representing the state, and uses it to look up the appropriate leaf in the Q-table,
 from which it returns the action, either by random selection or the one with the highest expected reward.

giveReward takes a float and adjusts the last leaf to reflect the reward it has recieved.

It is very important to call getAction and giveReward in alternating order, otherwise the agent will not learn properly.

I have taken an unconventional approach to encoding the Q-table. Instead of using a 2D list (table), I have used nested
 dicts in a sort of tree configuration. The advantage of this is that we do not have to 'set up' the Q-table on
 instantiating the class. The Q-tree will grow dynamically as new states are discovered, and until all states have been
 discovered. It also means that we don't need to translate to indices from states and vice versa, the states themselves
 can be used as keys to access policy, be they integers, floats or strings. See getLeaf() for how the tree is
 traversed and newLeaf() for how the tree grows upon discovery of a new state.

I have also switched the time-direction of updating expected rewards. Normally, when updating the Q-table, the agent will
 'look ahead' in the next state to see if its a poor position. Due to the stochastic nature of the system, it is
 difficult to predict the next state, even knowing the action taken in the current state. As a result, instead this agent
 will 'look back', it will remember the previouse state and action, and adjust the expected reward of that combination to 
 reflect the current reward. This will allow the agent to know if an action will lead to a poor position. See giveReward()
 for how the policies are updated.
'''
class QLearningAgent:
	
	'''
	actions is an list that gives the possible actions the agent can take
	discount controls how much the agent values future rewards, high discount values mean the agent 'plans ahead' more
	exploration controls how likely the agent is to randomly select policy
	'''
	def __init__(self, actions=[0,1], discount=0.25 , exploration=0.1):
		self.Q = {}
		self.actions = actions
		self.discount = discount
		self.exploration = exploration

		# These 4 variable are needed so that the agent can know which state/action combinations are likely to lead to one
		# another, and adjust policy accordingly.
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

	'''
	Creates a new branch in the Q-tree, will be called when the state has not been discovered yet and so there is no branch
	 on the Q-tree to record a policy for it.
	state is a list representing the state we need to create a new branch for. Note that it does not necessarily have to be
	 the same length as the state vector, as we can have new states where the first few state components are the same as in
	 states we have already visited.
	'''
	def newLeaf(self, state):
		if (len(state) > 0):
			# recursively assembles a new branch until there are no remaining elements in state
			return {state[0]: self.newLeaf(state[1:len(state)])}
		else:
			# if we have reached the end of the state list, we can now create a dict that will give both the expected reward
			#  for all available actions and also the alpha factor for each state/action combination
			q = {}
			for i in self.actions:
				# randomly initializes the expected reward, the alpha factor is initialized to one
				q[i] = {'reward':uniform(-1, 1), 'alpha':1}

			return q

	'''
	Recursively traverses the Q-tree, with exception handling for when the given state has not been discovered yet.
	state is an list that describes both the current state of the system and the path to where the Q-tree stores the
	 expected rewards for taking each action in that state. This is one of the advantages of using a Q-tree instead of a Q-table, the
	 state information itself can be used to access policy.
	q is a Q-tree. Note that the Q-tree stored in the variable self.Q is basically a dict with keys corresponding to
	 all the possible values of the first state component, and values being smaller Q-trees that contain policy for when
	 the first state component takes the corresponding value. This is why we can access the tree recursively.
	'''
	def getLeaf(self, state, q):
		if(len(state) > 0):
			# This block will execute if we have not yet reached the bottom of the Q-tree.
			# the basic idea is to use the first element of the state list to access a subsidiary Q-tree, and call
			# getLeaf on it. We must also remove the first element of state.
			try:
				# q[state[0]] will throw a KeyError exception if we have not yet discovered the given state, in which
				# case the except block below will execute, and create a new branch on the Qtree corresponding to state.
				return self.getLeaf(state[1:len(state)], q[state[0]])
			except(KeyError):
				# this block will execute if the given state has not been discovered yet.
				# simply create a new branch on the Q-tree corresponding to the given state, and continue normally.
				q[state[0]] = self.newLeaf(state[1:len(state)])
				return self.getLeaf(state[1:len(state)], q[state[0]])
		else:
			# If state is an empty list, we have hit the bottom of the tree and q is simply a dict with actions mapping
			#  to expected reward and alpha.
			return q

	'''
	This function will either look into the agent's Q-tree to find the action with the highest expected
	 reward, or it will simply return a random action.
	state is a list of state coordinates, this function will use it to find the policy it has learned for 
	 the given state.
	'''
	def getAction(self, state):
		# the agent has a small percent chance, given by self.exploration, of simply taking a random action
		if (uniform(0, 1) < self.exploration):
			# randomly selects an action
			action = choice(self.actions)
		else:
			# looks into its Q-tree to find the learned policy for the current state
			# This is a dict where actions map to secondary dicts with two elements, 'reward' and 'alpha'
			q = self.getLeaf(state, self.Q)

			# selects the action with the highest expected reward
			action = self.actions[0]
			for i in self.actions:
				if (q[i]['reward'] > q[action]['reward']):
					action = i

		# keeps track of the last 2 actions it has taken, this will be used in giveReward() to allow the 
		# agent to learn.
		self.secondLastState = self.lastState
		self.secondLastAction = self.lastAction
		self.lastState = state
		self.lastAction = action

		return action

	'''
	This function updates the policy in the Q-table to reflect rewards it is given as a parameter. It can only
	 remember the last state it used to pick an action, and so it must be called after getAction(), otherwise
	 the learning algorithm will not function properly.
	Reward is a float that gives the reward of the last state/action combination.
	'''
	def giveReward(self, reward):

		# gets the last used policy
		lastLeaf = self.getLeaf(self.lastState, self.Q)

		# changes the expected reward of the last state/action combination to a weighted average between the previous
		# expected reward and the current reward
		alpha = lastLeaf[self.lastAction]['alpha']
		beta = 1 - alpha
		# we need to store the update in a variable temp since we'll use it again in a bit
		temp = beta*lastLeaf[self.lastAction]['reward'] + alpha*reward
		lastLeaf[self.lastAction]['reward'] = temp

		# we now update the second last policy, using the same rule with different parameters.
		# This is so that the agent can think ahead, we don't want it to take an action with high reward that will
		# lead it into a bad position. We accomplish this by making the reward for a certain state/action combination
		# a weighted average of itself and the state/action combinations it is likely to lead to.
		secondlastLeaf = self.getLeaf(self.secondLastState, self.Q)
		alpha = secondlastLeaf[self.lastAction]['alpha']*self.discount
		secondlastLeaf[self.secondLastAction]['reward'] += alpha*temp 

		# updates the reducer factor of the last action so that if n is the number of times that state has been visited,
		# the reducer factor equals 1/n
		newDenominator = 1/lastLeaf[self.lastAction]['alpha'] + 1
		lastLeaf[self.lastAction]['alpha'] = 1/newDenominator

	def toString(self):
		return yaml.dump(self.__dict__)

	def fromString(self, string):
		# parses yaml into a config object
		config = yaml.safe_load(string)

		# uses config object to configure variables to match the saved model
		self.Q = config['Q']
		self.actions = config['actions']
		self.discount = config['discount']
		self.exploration = config['exploration']
		self.secondLastState = config['secondLastState']
		self.secondLastAction = config['secondLastAction']
		self.lastState = config['lastState']
		self.lastAction = config['lastAction']