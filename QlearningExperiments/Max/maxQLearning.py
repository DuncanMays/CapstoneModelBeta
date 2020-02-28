
from random import uniform, choice

class QLearningAgent:
	
	def __init__(self, actions=[0,1], beta=0.25, rho=0.1, delta = 0.1):
		self.Q = {}
		self.actions = actions
		self.beta = beta
		self.rho = rho

		# the tolerance of the agent to deviations from percieved optimality
		self.delta = delta
		# used to indicate weather or not the agent is currently learning
		self.exploring = False

		self.secondLastState = 'bootstrap'
		self.secondLastAction = 'bootstrap'
		self.lastState = 'bootstrap'
		self.lastAction = 'bootstrap'

		self.getAction(['bootstrap'])

	def newLeaf(self, state):
		if (len(state) > 0):
			return {state[0]: self.newLeaf(state[1:len(state)])}
		else:
			leaf = {}
			for i in self.actions:
				leaf[i] = {'reward':uniform(-1, 1), 'alpha':1}

			# policy is randomly initialized
			leaf['policy'] = choice(self.actions)
			# this indicates that the dict is indeed a leaf, rather than just another node in the Qtree
			leaf['type'] = 'leaf'

			return leaf

	def getLeaf(self, state, q):
		if(len(state) > 0):
			try:
				return self.getLeaf(state[1:len(state)], q[state[0]])
			except(KeyError):
				q[state[0]] = self.newLeaf(state[1:len(state)])
				return self.getLeaf(state[1:len(state)], q[state[0]])
		else:
			return q

	def getAction(self, state):
		if (uniform(0, 1) < self.rho):
			# small probability of selecting a random acion
			action = choice(self.actions)
		else:
			# simply follows policy
			action = self.getLeaf(state, self.Q)['policy']

		self.secondLastState = self.lastState
		self.secondLastAction = self.lastAction
		self.lastState = state
		self.lastAction = action

		return action

	def giveReward(self, reward):
		if(not self.exploring):
			return

		lastLeaf = self.getLeaf(self.lastState, self.Q)

		alpha = lastLeaf[self.lastAction]['alpha']
		beta = 1 - alpha
		temp = beta*lastLeaf[self.lastAction]['reward'] + alpha*reward
		lastLeaf[self.lastAction]['reward'] = temp

		secondlastLeaf = self.getLeaf(self.secondLastState, self.Q)
		alpha = secondlastLeaf[self.lastAction]['alpha']*self.beta
		secondlastLeaf[self.secondLastAction]['reward'] += alpha*temp 

		newDenominator = 1/lastLeaf[self.lastAction]['alpha'] + 1
		lastLeaf[self.lastAction]['alpha'] = 1/newDenominator

	# this method is called when the agent is finished exploring
	def finishExploring(self):
		self.exploring = False

		# updates the policy key in all leaves in the Qtree, it also resets
		#  all alpha values to 1 for every state/action combination.
		self.updatePolicy(self.Q)

	def updatePolicy(self, q):
		if(('type' in q) and (q['type'] == 'leaf')):
			# this block will execute if q is a leaf

			# iterates accross the actions for two reasons:
			#  the first is resets their alpha values to 1
			#  the second is to find the action with the maximal expected reward
			optimalAction = self.actions[0]
			for i in self.actions:
				# resets alpha
				q[i]['alpha'] = 1
				# checks if i has a greater expected reward than the optimal
				#  action, if it does, set optimalAction to i
				if (q[i]['reward'] > q[optimalAction]['reward']):
					optimalAction = i

			# this set will hold the actions that have expected reward within
			#  self.delta of the optimal action
			A = []
			for i in self.actions:
				if (q[i]['reward'] > q[optimalAction]['reward']-self.delta):
					A.append(i)

			# sets the policy for the state this leaf corresponds to to a random
			#  element from A
			q['policy'] = choice(A)

		else:
			# this block will execute if q is a node in the Qtree
			for i in q:
				# recursively calls updatePolicy on all branches off of q
				self.updatePolicy(q[i])

	# this method is called when the agent begins exploring
	def startExploring(self):
		self.exploring = True