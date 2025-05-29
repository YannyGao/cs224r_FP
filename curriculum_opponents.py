from baseline_agent import BaselineAgent
import random
import torch


class WeakOpponent:
    def get_action(self, obs, mask):
        valid_actions = [i for i, m in enumerate(mask) if m == 1]
        if not valid_actions:
            action = 0
        else:
            action = random.choice(valid_actions)
        # Dummy log_prob and value to match expected return
        return action, torch.tensor(0.0), torch.tensor(0.0)

    
class MediumOpponent:
    def __init__(self, base_agent):
        self.agent = base_agent

    def get_action(self, obs, mask):
        return self.agent.get_action(obs, mask)

def curriculum_schedule():
    return [
        (0, WeakOpponent()),
        (2000, MediumOpponent(BaselineAgent())),
        (6000, MediumOpponent(BaselineAgent())),
    ]

def select_opponent(ep, schedule):
    opponent = schedule[0][1]
    for start_ep, opp in schedule:
        if ep >= start_ep:
            opponent = opp
    return opponent
