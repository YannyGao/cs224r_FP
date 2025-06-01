# curriculum.py

from baseline_agent import BaselineAgent
import random
import torch


class WeakOpponent:
    def get_action(self, obs, mask):
        valid_actions = [i for i, m in enumerate(mask) if m == 1]
        action = random.choice(valid_actions) if valid_actions else 0
        # Dummy log_prob and value to match expected return structure
        return action, torch.tensor(0.0), torch.tensor(0.0)


class MediumOpponent:
    def __init__(self, base_agent: BaselineAgent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)


def curriculum_schedule():
    shared_agent = BaselineAgent()
    return [
        (0, WeakOpponent()),
        (2000, MediumOpponent(shared_agent)),
        (6000, MediumOpponent(shared_agent)),  # same agent can keep learning
    ]


def select_opponent(ep: int, schedule: list):
    opponent = schedule[0][1]
    for start_ep, opp in schedule:
        if ep >= start_ep:
            opponent = opp
    return opponent
