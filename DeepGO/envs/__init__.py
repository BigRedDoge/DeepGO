from gym.envs.registration import register

register(
    id='DeepGo-v0',
    entry_point='DeepGO.envs:DeepGOEnv',
)