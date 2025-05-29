# deception_reward.py

import torch
from treys import Deck, Evaluator, Card

evaluator = Evaluator()

def index_to_card(index):
    suits = ['s', 'h', 'd', 'c']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    if 0 <= index < 52:
        rank = ranks[index % 13]
        suit = suits[index // 13]
        return rank + suit
    return None

def decode_cards(obs):
    one_hot = obs[:52]
    indices = [i for i, val in enumerate(one_hot) if val == 1.0]
    cards = []
    for i in indices:
        card_str = index_to_card(i)
        if card_str is not None:
            try:
                cards.append(Card.new(card_str))
            except Exception as e:
                print(f"[Card Conversion Error] Index {i} → {card_str}: {e}")
    hole_cards = cards[:2]
    community_cards = cards[2:]
    return hole_cards, community_cards

def estimate_bluff_score(hole_cards, community_cards, num_simulations=20):
    wins = 0
    for _ in range(num_simulations):
        deck = Deck()
        used = set(hole_cards + community_cards)
        for c in used:
            deck.cards.remove(c)

        remaining = 5 - len(community_cards)
        board = community_cards + deck.draw(remaining)
        opponent = deck.draw(2)

        my_score = evaluator.evaluate(board, hole_cards)
        opp_score = evaluator.evaluate(board, opponent)
        if my_score < opp_score:
            wins += 1
    return wins / num_simulations

def compute_deception_reward(obs, action, final_reward, aggressive_threshold=0.8, aggressive_actions={2, 3, 4}):
    """
    Calculates deception reward if action was a bluff and resulted in success.
    """
    try:
        hole_cards, community_cards = decode_cards(obs)
        bluff_score = 1.0 - estimate_bluff_score(hole_cards, community_cards)
        if bluff_score < 0.3 and action in [2, 3, 4]:
            if final_reward > 0:  # Only reward successful bluffs
                print(f"[Bluff Detected] Score={bluff_score:.2f}, Action={action}, Final Reward={final_reward}")
                return 0.1

        if bluff_score >= aggressive_threshold and action in aggressive_actions and final_reward > 0:
            # Agent bluffed and succeeded
            
            return 0.2 * final_reward  # tunable shaping factor
    except Exception as e:
        print(f"[Deception Reward Error] {e}")
    return 0.0
