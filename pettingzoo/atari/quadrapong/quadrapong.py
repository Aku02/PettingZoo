import os

from ..base_atari_env import BaseAtariEnv, base_env_wrapper_fn, parallel_wrapper_fn
from glob import glob

def raw_env(**kwargs):
    mode = 33
    num_players = 4
    name = os.path.basename(__file__).split(".")[0]
    parent_file = glob('./pettingzoo/atari/' + name + '*.py')
    version_num = parent_file[0].split('_')[-1].split('.')[0]
    name = name + "_" + version_num
    return BaseAtariEnv(
        game="pong",
        num_players=num_players,
        mode_num=mode,
        env_name=name,
        **kwargs
    )


env = base_env_wrapper_fn(raw_env)
parallel_env = parallel_wrapper_fn(env)
