from gym.envs.registration import register

register(
    id='MEWASymbolic-v0',
    entry_point='mewa.mewa_symbolic:MEWASymbolic',
    kwargs={
        'task_path': None,
        'wide_tasks': 1,
        'narrow_tasks': 50,
        'complex_worker': True,
        'seed': None
    }
)

register(
    id='MEWASymbolicPrivileged-v0',
    entry_point='mewa.mewa_symbolic_privileged:MEWASymbolicPrivileged',
    kwargs={
        'task_path': None,
        'wide_tasks': 1,
        'narrow_tasks': 50,
        'complex_worker': True,
        'seed': None,
        'privilege_level': 3
    }
)

register(
    id='MEWASymbolicSingle-v0',
    entry_point='mewa.mewa_symbolic_single:MEWASymbolicSingle',
    kwargs={
        'task_path': None,
        'wide_tasks': None,
        'narrow_tasks': None,
        'complex_worker': None,
        'single_task_worker': None,
        'seed': None,
    }
)
