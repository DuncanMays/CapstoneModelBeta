from maxQLearning import QLearningAgent


agent = QLearningAgent()

agent.getAction([1,2,3])
agent.giveReward(3)

agent.startExploring()

agent.getAction([1,2,3])
agent.giveReward(3)

agent.finishExploring()

agent.getAction([1,2,3])
agent.giveReward(3)