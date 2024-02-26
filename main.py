from pettingzoo.classic import texas_holdem_no_limit_v6

env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=4)
env.reset()

def print_obs(obs):
    i = 0
    for item in observation['observation']:
        if i <= 12:
            print(f'spades {i} : '+str(item))
        elif i <= 25:
            print('hearts '+str(i % 13)+': '+str(item))
        elif i <= 38:
            print('diamonds '+str(i % 13)+': '+str(item))
        elif i <= 51:
            print('clubs '+str(i % 13)+': '+str(item))
        elif i <= 59:
            print('player '+str(i % 51)+' chips: '+str(item))
        elif i == 60:
            print('total pot: '+str(item))
        elif i <= 68:
            print('total bets player '+str(i % 60)+': '+str(item))
        i += 1
    print('----------------------------------------------------------------')

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print_obs(observation)

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)

    env.step(action)
env.close()
