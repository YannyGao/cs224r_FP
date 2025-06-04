def is_preflop(env):
    return env.unwrapped.get_stage() == "preflop"

def is_postflop(env):
    return env.unwrapped.get_stage() in ["flop", "turn", "river"]

def is_flop_card_revealed(state):
    # Community cards start at position 2 * 52
    # Very hacky, but if any non-zero in this slice: flop is revealed
    community_cards = state[2 * 52 : 2 * 52 + 5 * 52]
    return any(community_cards)
