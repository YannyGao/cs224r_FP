import matplotlib.pyplot as plt

episodes = [1000 * i for i in range(1, 11)]

agent0_scores = [1, 2, 4, 3, 3, 4, 2, 5, 3, 7]
agent1_scores = [0, 1, 0, 1, 0, -1, 0, 0, 0, 0]
agent2_scores = [1, 1, 0, 7, 1, -3, -4, -3, 6, -1]

plt.figure(figsize=(10, 6))
plt.plot(episodes, agent0_scores, label='Agent0 vs. Random', marker='o')
plt.plot(episodes, agent1_scores, label='Agent1 vs. Agent0', marker='s')
plt.plot(episodes, agent2_scores, label='Agent2 vs. Agent1', marker='^')

plt.title('Average Score per 1000 Episodes Across Agent Generations')
plt.xlabel('Episodes')
plt.ylabel('Average Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
