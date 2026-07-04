from gymnasium.envs.registration import register

from .ConnectFourEnv import ConnectFourEnv

register(
    id="Connect4-v0",
    entry_point=ConnectFourEnv,
)