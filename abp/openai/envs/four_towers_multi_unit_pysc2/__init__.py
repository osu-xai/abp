from gym.envs.registration import register

register(
    id='FourTowersSequential-v0',
    entry_point='four_towers_pysc2.FourTowersSequential:FourTowersSequential',
)