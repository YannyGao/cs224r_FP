import torch
import time
def tokenize_feedback(feedback_encoder, feedback: str) -> torch.Tensor:
    with torch.no_grad():
        embedding = feedback_encoder.encode(feedback, convert_to_tensor=True)
    return embedding.cpu()
def get_stage_from_obs(obs):
    card_vector = obs[:52]  # first 52 entries
    num_cards_total = int(sum(card_vector))  # total = player cards + community cards
    num_community_cards = num_cards_total - 2  # subtract 2 player hole cards

    if num_community_cards == 0:
        return "pre-flop"
    elif num_community_cards == 3:
        return "flop"
    elif num_community_cards == 4:
        return "turn"
    elif num_community_cards == 5:
        return "river"
    else:
        return "unknown"

def is_preflop(env):
    return env.unwrapped.get_stage() == "preflop"

def is_postflop(env):
    return env.unwrapped.get_stage() in ["flop", "turn", "river"]

def is_flop_card_revealed(state):
    # Community cards start at position 2 * 52
    # Very hacky, but if any non-zero in this slice: flop is revealed
    community_cards = state[2 * 52 : 2 * 52 + 5 * 52]
    return any(community_cards)

import openai




import requests
import time

MISTRAL_API_KEY = "3D2UaiLiM9BHoA6enEjzEdxOSQDLak23"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def get_adherence_score_from_llm(previous_feedback, stats_before, stats_after):
    """Get adherence score from LLM (-5 to 5 scale) using Mistral"""
    
    evaluation_prompt = f"""
    Previous strategic advice: "{previous_feedback}"
    
    Agent statistics BEFORE advice:
    - VPIP (hands played): {stats_before["VPIP"]:.1f}%
    - PFR (pre-flop raises): {stats_before["PFR"]:.1f}%
    - AFq (aggression frequency): {stats_before["AFq"]:.1f}%
    - WTSD (went to showdown): {stats_before["WTSD"]:.1f}%
    
    Agent statistics AFTER advice:
    - VPIP: {stats_after["VPIP"]:.1f}%
    - PFR: {stats_after["PFR"]:.1f}%
    - AFq: {stats_after["AFq"]:.1f}%
    - WTSD: {stats_after["WTSD"]:.1f}%
    
    
    How well did the agent follow the strategic advice? Rate adherence from -5 to 5. :
    
    Consider:
    - Direction of statistical changes (did they move toward advised targets?)
    - Magnitude of changes (how much did they adjust?)
    - Overall consistency with the strategic advice given
    - negative 5 is no adherence and 5 is perfect adherence to feedback.
    - give negative rewards if the new performance is not following advice.

    Respond with only a decimal number between -5 and 5.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-medium",  # or mistral-small / mistral-large
        "messages": [
            {"role": "system", "content": "You are a helpful and critical poker coach evaluating an RL agent."},
            {"role": "user", "content": evaluation_prompt.strip()}
        ],
        "temperature": 0.2,
        "max_tokens": 10
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"]
        score = float(message.strip())
        return max(-5, min(5.0, score))  # Clamp to [-5, 5]
    except Exception as e:
        time.sleep(60)
        print(f"Mistral adherence scoring failed: {e}")
        return 0.0  # fallback neutral

    # --- OLD GPT-based version (commented) ---
    # try:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful and critical poker coach evaluating an RL agent."},
    #             {"role": "user", "content": evaluation_prompt}
    #         ],
    #         temperature=0.2,
    #         max_tokens=10
    #     )
    #     score = float(response.choices[0].message.content.strip())
    #     return max(0.0, min(5.0, score))
    # except Exception as e:
    #     time.sleep(30)
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful and critical poker coach evaluating an RL agent."},
    #                 {"role": "user", "content": evaluation_prompt}
    #             ],
    #             temperature=0.1,
    #             max_tokens=10
    #         )
    #         score = float(response.choices[0].message.content.strip())
    #         return max(0.0, min(5.0, score))
    #     except Exception as e:
    #         print(e)
    #         print(f"LLM adherence scoring failed: {e}")
    #     return 0.5

def get_gpt_feedback(player_stats, opponent_stats):
    """Get strategic feedback from Mistral"""
    
    def fmt(stats):
        return "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in stats.items()])

    prompt = f"""
    You are a poker coach analyzing an RL agent's performance in a game against an opponent.

    Here are the agent's stats:
    {fmt(player_stats)}

    Here are the opponent's stats:
    {fmt(opponent_stats)}

    Please write a short descriptive 10-word feedback giving strategy suggestions to our playing agent. 
    Be specific and technical.
    Give specific suggestions (e.g., bluffing frequency, risk-taking, timing, loose/tight, aggressive/passive, etc).
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": "You are a helpful and critical poker coach evaluating an RL agent."},
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        time.sleep(60)
        print(f"Mistral feedback generation failed: {e}")
        return ""

    # --- OLD GPT-based version (commented) ---
    # try:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=format_feedback_prompt(player_stats, opponent_stats),
    #         temperature=0.7
    #     )
    # except Exception as e:
    #     time.sleep(30)
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=format_feedback_prompt(player_stats, opponent_stats),
    #             temperature=0.7
    #         )
    #     except Exception as e:
    #         print(e)
    #         return ""
    # return response['choices'][0]['message']['content']


# PettingZoo action mapping
FOLD = 0
CALL_CHECK = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4

def is_voluntary_fund(action):
    """Returns True if action is a voluntary preflop fund (any except fold)."""
    return action != FOLD

def is_raise(action):
    """Returns True if action is a raise."""
    return action in [RAISE_HALF_POT, RAISE_POT, ALL_IN]

def extract_stage_action_counts(hand_actions, stage):
    """
    hand_actions: list of (stage, action_id) tuples for a single player in a single hand
    Returns counts of raises, calls, folds at given stage.
    """
    raises = sum(1 for s, a in hand_actions if s == stage and is_raise(a))
    calls = sum(1 for s, a in hand_actions if s == stage and a == CALL_CHECK)
    folds = sum(1 for s, a in hand_actions if s == stage and a == FOLD)
    return raises, calls, folds

def player_voluntarily_funded(hand_actions):
    """Checks if player voluntarily funded (VPIP) preflop in this hand."""
    preflop_actions = [a for s, a in hand_actions if s == "preflop"]
    return any(is_voluntary_fund(a) for a in preflop_actions)

def calculate_poker_stats(all_hands_actions):
    """
    all_hands_actions: list of hands, each hand is a list of (stage, action) tuples for a single player.

    Returns dictionary of VPIP, PFR, AFq, WTSD percentages.
    """

    total_hands = len(all_hands_actions)
    hands_voluntarily_funded = 0
    preflop_raises = 0
    total_raises = 0
    total_calls = 0
    total_folds = 0
    hands_to_showdown = 0
    hands_to_flop = 0

    for hand in all_hands_actions:
        # VPIP count
        if player_voluntarily_funded(hand):
            hands_voluntarily_funded += 1

        # PFR count
        raises, calls, folds = extract_stage_action_counts(hand, "preflop")
        preflop_raises += raises

        # AFq counts (all stages)
        r, c, f = extract_stage_action_counts(hand, "preflop")
        total_raises += r
        total_calls += c
        total_folds += f
        r, c, f = extract_stage_action_counts(hand, "flop")
        total_raises += r
        total_calls += c
        total_folds += f
        r, c, f = extract_stage_action_counts(hand, "turn")
        total_raises += r
        total_calls += c
        total_folds += f
        r, c, f = extract_stage_action_counts(hand, "river")
        total_raises += r
        total_calls += c
        total_folds += f

        # WTSD: did hand go to flop?
        stages_in_hand = {s for s, _ in hand}
        if "flop" in stages_in_hand:
            hands_to_flop += 1
            # Did hand go to showdown? Assume we can detect it from presence of "showdown" stage
            if "showdown" in stages_in_hand:
                hands_to_showdown += 1

    VPIP = (hands_voluntarily_funded / total_hands) * 100 if total_hands > 0 else 0
    PFR = (preflop_raises / total_hands) * 100 if total_hands > 0 else 0
    AFq = (total_raises / (total_raises + total_calls + total_folds)) * 100 if (total_raises + total_calls + total_folds) > 0 else 0
    WTSD = (hands_to_showdown / hands_to_flop) * 100 if hands_to_flop > 0 else 0

    return {
        "VPIP": VPIP,
        "PFR": PFR,
        "AFq": AFq,
        "WTSD": WTSD,
    }


def init_stats_counters():
    return {
        "hands_dealt": 0,
        "hands_played_preflop": 0,  # VPIP numerator
        "hands_raised_preflop": 0,  # PFR numerator
        "total_actions": 0,
        "aggressive_actions": 0,  # raise counts for AFq numerator
        "called_or_folded_actions": 0,  # calls + folds for AFq denominator complement
        "hands_went_to_flop": 0,  # denominator for WTSD
        "hands_went_to_showdown": 0,  # numerator for WTSD
    }

def update_stats_on_hand(stats, preflop_actions, actions, went_to_showdown=None):
    """
    stats: dict, the stats counters for this opponent
    preflop_actions: list of actions during preflop (ints)
    actions: list of all actions in the hand (ints)
    went_to_showdown: bool, did this hand go to showdown?

    Updates stats counters accordingly.
    """

    stats["hands_dealt"] += 1

    # VPIP: hands played preflop = count of non-fold preflop actions
    if any(a != 0 for a in preflop_actions):  # fold = 0
        stats["hands_played_preflop"] += 1

    # PFR: hands raised preflop (actions 2,3,4) 
    if any(a in [2,3,4] for a in preflop_actions):
        stats["hands_raised_preflop"] += 1

    # AFq: aggressive action frequency = raise / (call+fold+raise)
    for a in actions:
        stats["total_actions"] += 1
        if a in [2, 3, 4]:
            stats["aggressive_actions"] += 1
        elif a in [0, 1]:
            stats["called_or_folded_actions"] += 1

    # WTSD: hands went to showdown / hands went to flop
    # if went_to_showdown:
    #     stats["hands_went_to_showdown"] += 1

    # We can say hands went to flop if community cards were revealed (3+ cards)
    # You need to detect this per hand in your environment and call update accordingly
    if True:  # update outside, pass in this info as parameter if needed
        stats["hands_went_to_flop"] += 1

def compute_stats(stats):
    """
    Computes VPIP, PFR, AFq, WTSD percentages from raw counters.
    Returns dict of percentages.
    """
    vpip = (stats["hands_played_preflop"] / stats["hands_dealt"] * 100) if stats["hands_dealt"] > 0 else 0
    pfr = (stats["hands_raised_preflop"] / stats["hands_dealt"] * 100) if stats["hands_dealt"] > 0 else 0
    denominator = stats["aggressive_actions"] + stats["called_or_folded_actions"]
    afq = (stats["aggressive_actions"] / denominator * 100) if denominator > 0 else 0
    wtsd = (stats["hands_went_to_showdown"] / stats["hands_went_to_flop"] * 100) if stats["hands_went_to_flop"] > 0 else 0

    return {
        "VPIP": round(vpip, 2),
        "PFR": round(pfr, 2),
        "AFq": round(afq, 2),
        "WTSD": round(wtsd, 2),
    }
