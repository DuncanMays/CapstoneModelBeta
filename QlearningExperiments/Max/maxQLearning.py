import numpy as np


class QLearningAgent:
	
	def __init__(self, actions=None, beta=0.25, rho=0.1, delta=0.1):

		if actions is None:
			actions = [0, 1]

		self.Q = {}

		# action space; disable specific actions in specific states in script
		self.actions = actions

		# discount factor; how valuable future is relative to present
		self.beta = beta

		# exploration probability
		self.rho = rho

		# tolerance for suboptimal actions
		self.delta = delta

		# keep track of current state
		self.state_current = ['bootstrap']
		self.action_current = ['bootstrap']

		# initialize
		self.getAction(['bootstrap'])

	# create a leaf corresponding to a specific state
	def newLeaf(self, state):

		# recursive call through state dimensions - building tree
		if len(state) > 0:
			return {state[0]: self.newLeaf(state[1:len(state)])}

		# leaf: Q-values, visit counts, action to take according to policy, and indicator that this is a leaf
		else:
			leaf = {}
			for i in self.actions:
				leaf[i] = {'Q': 0, 'alpha': 1}
			leaf['policy'] = np.random.choice(self.actions)
			leaf['type'] = 'leaf'
			return leaf

	# find a leaf corresponding to a specific state
	def getLeaf(self, state, q):

		# recursive call through state dimensions - find the leaf
		if len(state) > 0:

			# if leaf already exists
			try:
				return self.getLeaf(state[1:len(state)], q[state[0]])

			# if not
			except KeyError:
				q[state[0]] = self.newLeaf(state[1:len(state)])
				return self.getLeaf(state[1:len(state)], q[state[0]])

		# return the leaf
		else:
			return q

	# get an action based on current state
	def getAction(self, state):

		# if exploring, take a random action with probability rho
		if np.random.uniform(0, 1) < self.rho:
			action = np.random.choice(self.actions)

		# otherwise, just follow action indicated in policy
		else:
			action = self.getLeaf(state, self.Q)['policy']

		self.action_current = action
		return action

	# update Q-tree
	def updateQ(self, cost, new_state):

		next_leaf = self.getLeaf(new_state, self.Q)
		temp = []
		for action in self.actions:
			temp.append(next_leaf[action]['Q'])
		Q_min = np.amin(temp)

		current_leaf = self.getLeaf(self.state_current, self.Q)
		alpha = current_leaf[self.action_current]['alpha']
		update = (1 - alpha) * current_leaf[self.action_current]['Q']
		update += alpha * (cost + self.beta * Q_min)

		alpha = 1 / alpha
		alpha += 1
		alpha = 1 / alpha

		current_leaf[self.action_current]['Q'] = update
		current_leaf[self.action_current]['alpha'] = alpha

		self.state_current = new_state

	# update policy
	def updatePolicy(self, q):
		# if q is a leaf
		if 'type' in q and q['type'] == 'leaf':

			optimalAction = self.actions[0]
			for i in self.actions:
				# checks if i has a greater expected reward than the optimal
				#  action, if it does, set optimalAction to i
				if q[i]['Q'] < q[optimalAction]['Q']:
					optimalAction = i

			# this set will hold the actions that have expected reward within
			#  self.delta of the optimal action
			A = []
			for i in self.actions:
				if q[i]['Q'] < q[optimalAction]['Q'] + self.delta:
					A.append(i)

			# sets the policy for the state this leaf corresponds to to a random
			#  element from A
			q['policy'] = np.random.choice(A)

		# otherwise
		else:
			for i in q:
				# recursively call updatePolicy on all branches off of q
				self.updatePolicy(q[i])
