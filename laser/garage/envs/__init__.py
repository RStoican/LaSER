import numpy as np
from gymnasium.envs.registration import register

# MEWA
# ----------------------------------------
register(
    'MEWASymbolic-v0',
    entry_point='garage.envs.mewa.mewa_symbolic:MEWASymbolic',
    kwargs={'task_path': None, 'wide_tasks': 1, 'narrow_tasks': 10,
            'complex_worker': True, 'h': 100, 'seed': np.random.randint(0, 65536),
            'uniform_human': False}
)

register(
    'MEWATaskAblation-v0',
    entry_point='garage.envs.mewa.mewa_task_ablation:MEWATaskAblation',
    kwargs={'task_path': None, 'wide_tasks': 1, 'narrow_tasks': 2,
            'complex_worker': True, 'h': 100, 'seed': np.random.randint(0, 65536)}
)

register(
    'MEWACurated-v0',
    entry_point='garage.envs.mewa.mewa_curated:MEWACurated',
    kwargs={'complex_worker': True, 'h': 100, 'seed': np.random.randint(0, 65536), 'i': -1}
)

# Meta-World
# ----------------------------------------
register(
    'MetaWorldML1-v0',
    entry_point='garage.envs.metaworld.metaworld_ml1:MetaWorldML1',
    kwargs={'task_type': None, 'mode': None, 'given_tasks': None, 'h': None,
            'seed': np.random.randint(0, 65536)}
)

register(
    'MetaWorldML10-v0',
    entry_point='garage.envs.metaworld.metaworld_ml10:MetaWorldML10',
    kwargs={'task_type': None, 'mode': None, 'given_tasks': None, 'h': None,
            'seed': np.random.randint(0, 65536)}
)

# MuJoCo
# ----------------------------------------
register(
    'MuJoCoHopper-v5',
    entry_point='garage.envs.mujoco.hopper:Hopper',
    kwargs={'mode': None, 'h': 400, 'seed': np.random.randint(0, 65536)}
)
