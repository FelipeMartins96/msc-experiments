from pettingzoo.mpe import simple_spread_v2
from stable_baselines3 import DDPG
import supersuit as ss

env = simple_spread_v2.parallel_env(continuous_actions=True)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")
model = DDPG("MlpPolicy", env, train_freq=1)

model.learn(total_timesteps=100000)

model.save('pl')