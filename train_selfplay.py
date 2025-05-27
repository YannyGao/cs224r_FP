import torch
import copy
import os
from statistics import mean
from pettingzoo.classic import texas_holdem_no_limit_v6
from agent import Agent
from policy import Policy
from constants import (
    EPISODES, GPU, CPU,
    OBSERVATION, ACTION_MASK,
    OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE
)

SAVE_DIR = "models/"
DEVICE = torch.device(GPU if torch.cuda.is_available() else CPU)

def init_new_policy():
    return Policy(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE, HIDDEN_LAYER_SIZE)

def save_policy(policy, gen_name):
    torch.save(policy.state_dict(), os.path.join(SAVE_DIR, f"{gen_name}.pt"))

def load_policy(gen_name):
    policy = init_new_policy()
    policy.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{gen_name}.pt")))
    return policy

def generate_episode(env, rl_agent: Agent, ant_agent: Agent):
    env.reset()

    log_probs = []
    rewards = []

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()
        if termination or truncation:
            action = None
        else:
            mask = observation[ACTION_MASK]
            state = observation[OBSERVATION]
            if agent_name == rl_agent.get_name():
                action, log_prob = rl_agent.get_action(state, mask)
                log_probs.append(log_prob)
            elif ant_agent is not None:
                action, _ = ant_agent.get_action(state, mask)
            else:
                action = env.action_space(agent_name).sample(mask)

        env.step(action)

        if agent_name == rl_agent.get_name():
            rewards.append(reward)

    if log_probs and rewards:
        backprop_rewards = [rewards[-1] for _ in rewards]
        rl_agent.REINFORCE(log_probs, backprop_rewards)

    return sum(rewards)

def train_generation(gen_index: int, opponent_gen: str | None):
    alpha = 0.01
    gamma = 0.99
    num_episodes = 10000

    env = texas_holdem_no_limit_v6.env(render_mode="ansi", num_players=2)
    print(f"\n--- Training Agent{gen_index} vs. {opponent_gen or 'random'} ---")

    rl_policy = init_new_policy()
    rl_agent = Agent(alpha, gamma, rl_policy, name="player_0")

    if opponent_gen:
        ant_policy = load_policy(opponent_gen)
        ant_agent = Agent(alpha, gamma, ant_policy, name="player_1")
    else:
        ant_agent = None

    score_batch = []
    for episode in range(1, num_episodes + 1):
        score = generate_episode(env, rl_agent, ant_agent)
        score_batch.append(score)

        if episode % 1000 == 0:
            avg_score = mean(score_batch)
            print(f"[Agent{gen_index}] Episode {episode}, Avg Score: {avg_score:.3f}")
            score_batch.clear()

    save_policy(rl_policy, f"agent{gen_index}")
    print(f"Agent{gen_index} saved.\n")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_generation(gen_index=0, opponent_gen=None)        # Train Agent0 vs. random
    train_generation(gen_index=1, opponent_gen="agent0")    # Train Agent1 vs. Agent0
    train_generation(gen_index=2, opponent_gen="agent1")    # Train Agent2 vs. Agent1

if __name__ == "__main__":
    main()
