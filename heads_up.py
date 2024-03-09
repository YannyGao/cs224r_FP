import torch
import copy
from statistics import mean
from pettingzoo.classic import texas_holdem_no_limit_v6

from agent import Agent
from policy import Policy
from constants import EPISODES, GPU, CPU, OBSERVATION, ACTION_MASK, OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE
from plot import plot_graph, plot_hist


# RENDER_MODE = "human"
RENDER_MODE = "rgb_array"

ACTION_MAP = [
    "Fold",
    "Check & Call",
    "Raise Half Pot",
    "Raise Full Pot",
    "All In"
]


def print_action(action: int | None, agent: str):
    if action != None:
        print(f"{agent} chooses {ACTION_MAP[action]}")
    else:
        print(f"{agent} finishes")


def generate_episode(rl_agent: Agent, sp_agent: Agent | None, action_data_ptr: list[int]):
    env = texas_holdem_no_limit_v6.env(render_mode=RENDER_MODE)
    env.reset()

    log_probs = []
    rewards = []

    for agent in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()
        if agent == rl_agent.get_name():
            rewards.append(reward)

        if termination or truncation:
            action = None
        else:
            mask = observation[ACTION_MASK]
            state = observation[OBSERVATION]
            if agent == rl_agent.get_name():
                action, log_prob = rl_agent.get_action(state, mask)

                log_probs.append(log_prob)
                action_data_ptr.append(action)
            elif sp_agent:
                action, _ = sp_agent.get_action(state, mask)
            else:
                action = env.action_space(agent).sample(mask)
        # print_action(agent, action)
        env.step(action)
    env.close()

    if log_probs:
        rl_agent.REINFORCE(log_probs, rewards)

    return sum(rewards)


def main():
    device = torch.device(GPU if torch.cuda.is_available() else CPU)
    print(f"Training using {device}")

    alpha = 0.01
    gamma = 0.9

    episodes = []
    scores = []
    score_batch = []
    action_data = []

    policy = Policy(
        OBSERVATION_SPACE_SIZE,
        ACTION_SPACE_SIZE,
        HIDDEN_LAYER_SIZE
    )

    previous_agent = None

    for episode in range(1, EPISODES + 1):
        agent = Agent(alpha, gamma, policy)

        score = generate_episode(agent, previous_agent, action_data)
        score_batch.append(score)

        if episode % 1000 == 0:
            avg_score = mean(score_batch)
            score_batch.clear()

            scores.append(avg_score)
            episodes.append(episode)

            previous_agent = copy.deepcopy(agent)

            print(f"EPISODE {episode}: {avg_score}")

    plot_hist(action_data, ACTION_SPACE_SIZE)
    plot_graph(
        'No-limit Texas Hold\'em: Score over episodes',
        "Episode",
        episodes,
        "Score",
        scores
    )


if __name__ == "__main__":
    main()
