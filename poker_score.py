from treys import Evaluator, Card, Deck
import random

class BluffScorer:
    def __init__(self, num_simulations=1000):
        self.evaluator = Evaluator()
        self.num_simulations = num_simulations

    def calculate_bluff_score(self, hole_cards, community_cards):
        """
        Estimate the probability of winning given your hand by sampling opponent hands and board completions.
        Returns a bluff score between 0 and 1, where a lower value suggests a higher risk bluff.
        """
        win_count = 0
        total_count = 0
        deck = Deck()

        # Remove known cards from deck
        known_cards = hole_cards + community_cards
        for card in known_cards:
            deck.cards.remove(card)

        for _ in range(self.num_simulations):
            sim_deck = deck.cards[:]
            random.shuffle(sim_deck)

            # Sample opponent hand
            opp_hole = [sim_deck.pop(), sim_deck.pop()]
            # Complete community cards to 5
            needed = 5 - len(community_cards)
            sim_community = community_cards + [sim_deck.pop() for _ in range(needed)]

            your_score = self.evaluator.evaluate(hole_cards, sim_community)
            opp_score = self.evaluator.evaluate(opp_hole, sim_community)

            if your_score < opp_score:  # Lower score is better
                win_count += 1
            total_count += 1

        bluff_score = 1.0 - (win_count / total_count)
        return bluff_score


# Example usage:
if __name__ == "__main__":
    hole = [Card.new('As'), Card.new('Ks')]  # Your hand
    community = [Card.new('2h'), Card.new('7d'), Card.new('Tc')]  # Flop

    scorer = BluffScorer(num_simulations=500)
    score = scorer.calculate_bluff_score(hole, community)
    print(f"Bluff Score: {score:.4f}")
