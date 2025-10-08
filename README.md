
# Bluffing with Precision: LLM-Guided Strategy and Opponent Modeling in Multi-Agent Poker

## Project Deliverable
- [Poster](cs224rposter.pdf)
- [Final Report](cs224rFinalProject-3.pdf)

## Motivation
Poker is a game where players must make decisions without knowing all information,
requiring strategic thinking and deception. Current AI poker systems like DeepStack work well but
cannot adapt to new opponents in real time or learn to bluff like humans do. We want to build AI
agents that can learn to bluff effectively, predict opponent behavior, and improve their strategy based
on feedback. Our main question is: Can AI agents learn to bluff strategically through reinforcement
learning combined with opponent modeling and language-based feedback?

## Method
We built a training system with four main parts. First, we use curriculum learning where
the AI practices against increasingly difficult opponents - starting with random players, then medium-
skilled bots, and finally strong learning agents. Second, we added a transformer model that tries to
predict what the opponent will do next based on their past actions. Third, we created a reward system
that gives extra points when the AI successfully bluffs (bets aggressively with weak cards and wins).
Fourth, we use an LLM (Mistral Medium) to analyze the AI’s play style using poker statistics like
VPIP and PFR, then give strategic advice that gets embedded into the AI’s decision-making process.

## Implementation 
We used the PettingZoo Texas Hold’em environment for two-player poker games.
The main AI agent uses a two-layer neural network trained with Actor-Critic reinforcement learning.
The opponent modeling component is a separate transformer that runs alongside the main agent and
adds its predictions to the observation space. We detect bluffs by running Monte Carlo simulations to
estimate hand strength - if the agent bets aggressively with weak estimated hands, we classify it as
a bluff. The LLM feedback system calculates poker statistics every episode, sends them to Mistral
Medium for strategic advice, and embeds the feedback using sentence transformers. We also tested
adding the LLM’s adherence scores as additional rewards.

## Results 
Our experiments show clear improvements across multiple metrics. Bluff reward tuning
found that intermediate reward values (around 2.0) work best, achieving 80.5% win rate and 20.9%
bluff success rate against strong opponents. Opponent modeling increased win rates against strong
opponents by 56.69% and improved bluff success rates significantly, though prediction accuracy
varied between runs. Curriculum learning outperformed static training across all opponent levels,
with the curriculum-trained agent achieving 65.2% win rate compared to 52.4% for the best static
approach. LLM feedback integration led to more disciplined play, reducing aggressive metrics like
AFq from 80% to 27%, and improved win rates against strong opponents by 20.65%. However, the
adherence reward mechanism failed completely, eliminating bluffing behavior entirely.

## Discussion 
The results show that our approach successfully teaches AI agents strategic deception
and adaptation. The bluff reward system encourages calculated risk-taking rather than random
aggression. Opponent modeling enables dynamic strategy adjustment based on predicted opponent
behavior. Curriculum learning builds robust strategies that transfer well to new situations. LLM
feedback helps refine play style using professional poker concepts. However, the adherence reward
failure highlights challenges in combining natural language feedback with reinforcement learning -
poker’s randomness makes strategic advice from previous games less relevant to new situations. The
wide variance in opponent modeling accuracy also suggests this component needs more investigation.

## Conclusion 
We demonstrated that AI agents can learn sophisticated poker strategies through behav-
ioral incentives, opponent modeling, and structured training progression. Our system produces agents
capable of strategic bluffing and real-time adaptation to opponent behavior. Key findings include:
optimal bluff rewards encourage effective deception, opponent modeling enables strategic adaptation,
curriculum learning beats static training, and LLM feedback can guide strategic development when
properly integrated. Future work should focus on more robust adherence mechanisms, improved
opponent modeling consistency, and applications to other strategic domains involving deception and
negotiation.
