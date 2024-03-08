import torch
import matplotlib.pyplot as plt

from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import Agent
from policy import Policy
from constants import GPU, CPU, OBSERVATION, ACTION_MASK

EPISODE_COUNT = 10

# TODO: Randomize seed
SEED = 42

# RENDER_MODE = "human"
RENDER_MODE = "rgb_array"


def generate_episode(rl_agent: Agent):
    env = texas_holdem_no_limit_v6.env(render_mode=RENDER_MODE)
    env.reset(seed=SEED)
    torch.manual_seed(SEED)

    log_probs = []
    rewards = []
    steps = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()

        if termination or truncation:
            action = None
            if (agent == rl_agent.get_name()):
                rewards.append(reward)
        else:
            mask = observation[ACTION_MASK]
            if (agent == rl_agent.get_name()):
                # state = observation[OBSERVATION]
                action, log_prob = rl_agent.get_action(observation[OBSERVATION], mask)
                log_probs.append(log_prob)
                rewards.append(reward)
                steps += 1
            else:
                action = env.action_space(agent).sample(mask)
        env.step(action)
    env.close()

    if (log_probs):
        rl_agent.REINFORCE(steps, log_probs, rewards)
    return sum(rewards)


def main():
    device = torch.device(GPU if torch.cuda.is_available() else CPU)
    print(f"Training using {device}")
    scores = []
    episodes = []
    policy = Policy()
    agent = Agent(0.01, 0.99, policy)
    for i in range(EPISODE_COUNT):
        score = generate_episode(agent)
        scores.append(score)
        episodes.append(i)
        print("SCORE EP ", i, ": ", score)
    plt.plot(episodes, scores)
    plt.xlabel('episode #')
    plt.ylabel('score')
    plt.title('Poker')
    plt.show()

if __name__ == "__main__":
    main()
