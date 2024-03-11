import torch
import copy
from statistics import mean
from pettingzoo.classic import texas_holdem_no_limit_v6

from agent import Agent
from policy import Policy
from constants import EPISODES, GPU, CPU, OBSERVATION, ACTION_MASK, OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE
from plot import plot_graph, plot_hist


RENDER_MODE = "ansi"


def generate_episode(env, rl_agent: Agent, ant_agent: Agent | None, action_data_ptr: list[int]):
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
            elif ant_agent:
                action, _ = ant_agent.get_action(state, mask)
            else:
                action = env.action_space(agent).sample(mask)

        env.step(action)
    env.close()

    rewards.pop(0)

    if log_probs and rewards:
        backprop_rewards = [rewards[-1] for _ in rewards]
        rl_agent.REINFORCE(log_probs, backprop_rewards)

    return sum(rewards)


def init_new_policy():
    return Policy(
        OBSERVATION_SPACE_SIZE,
        ACTION_SPACE_SIZE,
        HIDDEN_LAYER_SIZE
    )


def main():
    device = torch.device(GPU if torch.cuda.is_available() else CPU)
    print(f"Training using {device}")

    alpha = 0.01
    gamma = 0.99

    hands = 30

    episodes = []
    scores = []
    score_batch = []
    action_data = []

    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)

    policy = init_new_policy()
    ant_agent = None

    for episode in range(1, EPISODES + 1):
        agent = Agent(alpha, gamma, policy)

        score = generate_episode(env, agent, ant_agent, action_data)
        score_batch.append(score)

        if episode % 1000 == 0:
            avg_score = mean(score_batch)
            score_batch.clear()

            scores.append(avg_score)
            episodes.append(episode)

            ant_agent = copy.deepcopy(agent)

            print(f"EPISODE {episode}: {avg_score}")
            print(f"Fold: {action_data.count(0)}, Check/Call: {action_data.count(1)}, Raise Half: {action_data.count(2)}, Raise Full: {action_data.count(3)}, All: {action_data.count(4)}")

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
