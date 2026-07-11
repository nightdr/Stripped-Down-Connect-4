from gymnasium.envs.registration import register

from .ConnectFourEnv import ConnectFourEnv

register(
    id="ConnectFour-v0",
    entry_point=ConnectFourEnv,
)