from QLearningAgent import QLearningAgent

agent = QLearningAgent([1,2,3])

agent.getAction([123])
agent.giveReward(2)
agent.getAction([324])
agent.giveReward(43)
agent.getAction([324])
agent.giveReward(43)

cereal = agent.toString()

newAgent = QLearningAgent()
newAgent.fromString(cereal)

print(agent.__dict__)
print(newAgent.__dict__)