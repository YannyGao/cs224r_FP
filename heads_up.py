import torch

from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import Agent
from policy import Policy
from constants import EPISODES, GPU, CPU, OBSERVATION, ACTION_MASK, ACTION_SPACE_SIZE
from plot import plot_graph, plot_hist


# RENDER_MODE = "human"
RENDER_MODE = "rgb_array"


def generate_episode(rl_agent: Agent, action_data_ptr: list[int]):
    env = texas_holdem_no_limit_v6.env(render_mode=RENDER_MODE)
    env.reset()

    log_probs = []
    rewards = []

    for agent in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()

        if termination or truncation:
            action = None
            if (agent == rl_agent.get_name()):
                rewards.append(reward)
        else:
            mask = observation[ACTION_MASK]
            if (agent == rl_agent.get_name()):
                state = observation[OBSERVATION]
                action, log_prob = rl_agent.get_action(state, mask)
                action_data_ptr.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
            else:
                action = env.action_space(agent).sample(mask)
        env.step(action)
    env.close()

    if (log_probs):
        rl_agent.REINFORCE(log_probs, rewards)

    return sum(rewards)


def main():
    device = torch.device(GPU if torch.cuda.is_available() else CPU)
    print(f"Training using {device}")

    episodes = []
    scores = []
    action_data = []

    agent = Agent(0.01, 0.99, Policy())

    for episode in range(EPISODES):
        score = generate_episode(agent, action_data)
        print(f"EPISODE {episode}: {score}")
        scores.append(score)
        episodes.append(episode + 1)

    plot_hist(action_data, ACTION_SPACE_SIZE)
    plot_graph(
        'No-limit Texas Hold\'em: Score over episode',
        "Episode",
        episodes,
        "Score",
        scores
    )


if __name__ == "__main__":
    main()
