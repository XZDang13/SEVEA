HAMMER = {
    "env_name": "hammer-v3-goal-observable",
    "state_dim": 39
}

DRAWEROPEN = {
    "env_name": "drawer-open-v3-goal-observable",
    "state_dim": 39
}

PICKPLACE = {
    "env_name": "pick-place-v3-goal-observable",
    "state_dim": 39
}

PUSHBACK = {
    "env_name": "push-back-v3-goal-observable",
    "state_dim": 39
}

WINDOWOPEN = {
    "env_name": "window-open-v3-goal-observable",
    "state_dim": 39
}

QUADEUPED_WALK = {
    "domain_name": "quadruped",
    "task_name": "walk",
    "state_dim": 78,
    "camera":2
}

REACHER_HARD = {
    "domain_name": "reacher",
    "task_name": "hard",
    "state_dim": 6,
    "camera":0
}

HOPPER_HOP = {
    "domain_name": "hopper",
    "task_name": "stand",
    "state_dim": 15,
    "camera":0
}

WALKER_STAND = {
    "domain_name": "walker",
    "task_name": "walk",
    "state_dim": 24,
    "camera":0
}

FINGER_TURN = {
    "domain_name": "finger",
    "task_name": "turn_hard",
    "state_dim": 12,
    "camera":0
}

METAWORLD_CFGS = {
    "hammer": HAMMER,
    "draweropen": DRAWEROPEN,
    "pickplace": PICKPLACE,
    "windowopen": WINDOWOPEN,
    "pushback": PUSHBACK
}

DMC_CFGS = {
    "quadruped": QUADEUPED_WALK,
    "reacher": REACHER_HARD,
    "hopper": HOPPER_HOP,
    "walker": WALKER_STAND,
    "finger": FINGER_TURN
}
