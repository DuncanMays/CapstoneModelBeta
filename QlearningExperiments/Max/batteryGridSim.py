from maxQLearning import QLearningAgent
import numpy as np

# the actions that the agent will be allowed to perform
actions = np.linspace(-2, 2, 5, dtype=int)

# the state dimensions
time_step = 1
time = np.arange(0, 24, time_step)
localDemand = np.arange(0, 3)
priceOfProduction = np.arange(0, 3) * time_step
charge = np.arange(0, 5) - 2

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

numIterations = 30000
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

    # update policies every 1000 time steps
    if (i + 1) % 1000 == 0:
        for j in range(n_agents):
            agents[j].updatePolicy(agents[j].Q)

while 1:
    a = 0
