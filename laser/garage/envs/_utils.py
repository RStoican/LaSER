import gymnasium as gym
from laser.garage.envs.akro_replace.box import Box


def gym_to_akro(env):
    env.observation_space = from_gym(env.observation_space)
    # env.action_space = from_gym(env.action_space)
    return env


def from_gym(space):
    """Convert a gym.space to an akro.space.

    Args:
        space(:obj:`gym.Space`): The Space object to convert.

    Returns:
        akro.Space: The gym.Space object converted to an
            akro.Space object.

    """
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    else:
        raise TypeError(f'Unexpected space type: {type(space)}')
