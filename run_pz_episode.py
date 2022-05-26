from pettingzoo.mpe import simple_spread_v2
env = simple_spread_v2.env()

env.reset()

def policy(a, o):
    return env.action_space(a).sample()

for agent in env.agent_iter():
    env.render()
    observation, reward, done, info = env.last()
    action = policy(agent, observation) if not done else None
    env.step(action)