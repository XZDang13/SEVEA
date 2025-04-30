HAMMER = {
    "env_name": "hammer-v2-goal-observable",
    "state_dim": 39
}

DRAWEROPEN = {
    "env_name": "drawer-open-v2-goal-observable",
    "state_dim": 39
}

PICKPLACE = {
    "env_name": "pick-place-v2-goal-observable",
    "state_dim": 39
}

PUSHBACK = {
    "env_name": "push-back-v2-goal-observable",
    "state_dim": 39
}

WINDOWOPEN = {
    "env_name": "window-open-v2-goal-observable",
    "state_dim": 39
}

CHEETAH_RUN = {
    "domain_name": "cheetah",
    "task_name": "run",
    "state_dim": 17
}

REACHER_HARD = {
    "domain_name": "reacher",
    "task_name": "hard",
    "state_dim": 6
}

HOPPER_HOP = {
    "domain_name": "hopper",
    "task_name": "stand",
    "state_dim": 15
}

WALKER_STAND = {
    "domain_name": "walker",
    "task_name": "walk",
    "state_dim": 24
}

FINGER_TURN = {
    "domain_name": "finger",
    "task_name": "turn_hard",
    "state_dim": 12
}

METAWORLD_CFGS = {
    "hammer": HAMMER,
    "draweropen": DRAWEROPEN,
    "pickplace": PICKPLACE,
    "windowopen": WINDOWOPEN,
    "pushback": PUSHBACK
}

DMC_CFGS = {
    "cheetah": CHEETAH_RUN,
    "reacher": REACHER_HARD,
    "hopper": HOPPER_HOP,
    "walker": WALKER_STAND,
    "finger": FINGER_TURN
}