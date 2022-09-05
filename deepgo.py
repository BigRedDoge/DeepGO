from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3 import DQN, A2C, DDPG, PPO, TD3, SAC
import gym

from DeepGO.envs.deepgo_env import DeepGOEnv


env = make_vec_env(DeepGOEnv)

model = A2C(CnnPolicy, env, verbose=1, n_steps=5)

model.learn(total_timesteps=1000)

obs = env.reset()
env.render()
n_steps = 1000
for step in range(n_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

