from maxQLearning import QLearningAgent
import numpy as np

# the actions that the agent will be allowed to perform
actions = np.linspace(-2, 2, 5, dtype=int)

# the state dimensions
time_step = 1
time = np.arange(0, 24, time_step)
localDemand = np.arange(0, 3)
priceOfProduction = np.arange(0, 3) * time_step
charge = np.arange(0, 3)

priceOfRetail = [0.065, 0.094, 0.132] * time_step


class QLearningAgent(QLearningAgent):

    def getAction(self, state):

        action = super(QLearningAgent, self).getAction(state)

        if state != ['bootstrap']:
            # no pushing more than current charge
            if state[-1] + action < 0:
                action = -1 * state[-1]

            # no pulling more than difference between current charge and max
            elif state[-1] + action > charge[-1]:
                action = charge[-1] - state[-1]

            return action

    def updatePolicy(self, q, state=None):
        if state is None:
            state = []

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

            if state != ['bootstrap']:
                if state[-1] + q['policy'] < 0:
                    q['policy'] = -1 * state[-1]

                # no pulling more than difference between current charge and max
                elif state[-1] + q['policy'] > charge[-1]:
                    q['policy'] = charge[-1] - state[-1]

        # otherwise
        else:
            for i in q:
                # recursively call updatePolicy on all branches off of q
                tmp = state
                tmp.append(i)
                self.updatePolicy(q[i], tmp)


agents = []
n_agents = 5
for i in range(n_agents):
    agents.append(QLearningAgent(actions, delta=0.05))

# the state variables are randomly initialized
timeIndex = np.random.randint(len(time))
localDemandIndex = np.random.randint(len(localDemand), size=n_agents)
priceOfProductionIndex = np.random.randint(len(priceOfProduction))
chargeIndex = np.random.randint(len(charge), size=n_agents)

if 10 < time[timeIndex] < 17:
    priceOfRetailIndex = 2
elif 6 < time[timeIndex] < 19:
    priceOfRetailIndex = 1
else:
    priceOfRetailIndex = 0

numIterations = 300000
for i in range(numIterations):

    actions = []
    # get action from each agent
    for j in range(n_agents):
        actions.append(agents[j].getAction([time[timeIndex],
                                            localDemand[localDemandIndex[j]],
                                            priceOfProduction[priceOfProductionIndex],
                                            charge[chargeIndex[j]]]))

    # get immediate costs of actions
    costs = []
    for j in range(n_agents):
        cost = -priceOfRetail[priceOfRetailIndex] * localDemand[localDemandIndex[j]]
        if actions[j] < 0:
            cost += priceOfRetail[priceOfRetailIndex] * actions[j]
        elif actions[j] > 0:
            cost += priceOfProduction[priceOfProductionIndex] * actions[j]
        costs.append(cost)

    # update state
    chargeIndex = chargeIndex + actions

    timeIndex = (timeIndex + 1) % len(time)
    if 10 < time[timeIndex] < 17:
        priceOfRetailIndex = 2
    elif 6 < time[timeIndex] < 19:
        priceOfRetailIndex = 1
    else:
        priceOfRetailIndex = 0

    # update Q-tree
    for j in range(n_agents):
        state = [time[timeIndex],
                 localDemand[localDemandIndex[j]],
                 priceOfProduction[priceOfProductionIndex],
                 charge[chargeIndex[j]]]
        agents[j].updateQ(costs[j], state)

    # update policies every 10000 time steps
    if (i + 1) % 10000 == 0:
        for j in range(n_agents):
            agents[j].updatePolicy(agents[j].Q)

# for debugging
while 1:
    a = 0
