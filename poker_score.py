from treys import Evaluator, Card, Deck
import random

class BluffScorer:
    def __init__(self, num_simulations=1000):
        self.evaluator = Evaluator()
        self.num_simulations = num_simulations

    def calculate_bluff_score(self, hole_cards, community_cards):
        """
        Monte Carlo estimate: what is the probability we LOSE if we reach showdown?
        A high bluff score (closer to 1.0) means we likely lose → more incentive to bluff.
        """
        win_count = 0
        total_count = 0

        base_deck = Deck()
        for card in hole_cards + community_cards:
            base_deck.cards.remove(card)

        for _ in range(self.num_simulations):
            # Create a fresh simulation deck
            sim_deck = base_deck.cards[:]
            random.shuffle(sim_deck)

            # Sample opponent hole cards
            opp_hole = [sim_deck.pop(), sim_deck.pop()]

            # Sample rest of community cards (e.g., after the flop)
            num_needed = 5 - len(community_cards)
            sim_board = community_cards + [sim_deck.pop() for _ in range(num_needed)]

            my_score = self.evaluator.evaluate(sim_board, hole_cards)
            opp_score = self.evaluator.evaluate(sim_board, opp_hole)

            if my_score < opp_score:
                win_count += 1
            total_count += 1

        # Bluff score: how likely are we to lose at showdown
        bluff_score = 1.0 - (win_count / total_count)
        return bluff_score
