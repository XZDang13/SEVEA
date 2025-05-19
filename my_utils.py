import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time
    
def compute_distance(pos_from, pos_to, offset=None, plane_index=None):
    pos_from = np.array(pos_from)
    pos_to = np.array(pos_to)

    if offset is not None:
        pos_to = pos_to + np.array(offset)

    if plane_index is not None:
        pos_from = pos_from[..., plane_index]
        pos_to = pos_to[..., plane_index]

    distance = np.linalg.norm(pos_from - pos_to, axis=-1)

    return distance

def compute_direction(pos_from, pos_to, offset=None, plane_index=None):
    pos_from = np.array(pos_from)
    pos_to = np.array(pos_to)

    if offset is not None:
        pos_to = pos_to + np.array(offset)  # Create a new array to avoid modifying inputs

    if plane_index is not None:
        pos_from = pos_from[..., plane_index]  # Proper slicing for multi-dimensional inputs
        pos_to = pos_to[..., plane_index]

    direction = pos_to - pos_from
    norm = np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8  # Prevent division by zero
    direction /= norm  

    return direction

def compute_similarity(vec_1, vec_2):
    return np.dot(vec_1, vec_2)

def get_xpos(obs):
    return {
            "gripper": obs[:3],
            "finger_distance": obs[3],
            "object_1": obs[4:7],
            "object_2": obs[11:14],
            "goal": obs[-3:]
        }

class DetectMotion:
    @staticmethod
    def is_aligned(xpos, source_object, target_object, threshold, plane_index=[0, 1], offset=None):
        plane_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=plane_index, offset=offset)
        aligned = plane_distance <= threshold

        return aligned

    @staticmethod
    def is_above(xpos, source_object, target_object, offset=0, aligned=False):
        xy_index = [0, 1]
        z_index = 2
        xy_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=xy_index)

        is_xy_aligned = xy_distance <= 0.05
        is_above = (xpos[source_object][z_index] > (xpos[target_object][z_index]+offset)) and (xpos[source_object][z_index] <= (xpos[target_object][z_index]+offset+0.05))

        if aligned:
            return is_xy_aligned and is_above

        return is_above
    
    @staticmethod
    def is_below(xpos, source_object, target_object, offset=0, aligned=False):
        xy_index = [0, 1]
        z_index = 2
        xy_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=xy_index)

        is_xy_aligned = xy_distance <= 0.025
        is_below = xpos[source_object][z_index] < (xpos[target_object][z_index] + offset)

        if aligned:
            return is_xy_aligned and is_below
        
        return is_below
    
    @staticmethod
    def is_left(xpos, source_object, target_object, offset=0, aligned=False):
        yz_index = [1, 2]
        x_index = 0
        yz_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=yz_index)

        is_yz_aligned = yz_distance <= 0.05
        is_left = xpos[source_object][x_index] < (xpos[target_object][x_index] + offset)

        if aligned:
            return is_yz_aligned and is_left
        
        return is_left
    
    @staticmethod
    def is_right(xpos, source_object, target_object, offset=0, aligned=False):
        yz_index = [1, 2]
        x_index = 0
        yz_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=yz_index)

        is_yz_aligned = yz_distance <= 0.05
        is_right = xpos[source_object][x_index] > (xpos[target_object][x_index] + offset)

        if aligned:
            return is_yz_aligned and is_right

        return is_right

    @staticmethod
    def is_front(xpos, source_object, target_object, offset=0, aligned=False):
        xz_index = [0, 2]
        y_index = 1
        xz_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=xz_index)

        is_yz_aligned = xz_distance <= 0.05
        is_front = xpos[source_object][y_index] < (xpos[target_object][y_index] + offset)

        if aligned:
            return is_yz_aligned and is_front

        return is_front

    @staticmethod
    def is_back(xpos, source_object, target_object, offset=0, aligned=False):
        xz_index = [0, 2]
        y_index = 1
        xz_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=xz_index)

        is_yz_aligned = xz_distance <= 0.05
        is_back = xpos[source_object][y_index] > (xpos[target_object][y_index] + offset)

        if aligned:
            return is_yz_aligned and is_back

        return is_back

    @staticmethod
    def is_moving_to(xpos, next_xpos, source_object, target_object, threshold, offset=None):
        move_direction = compute_direction(xpos[source_object], next_xpos[source_object])
        goal_direction = compute_direction(xpos[source_object], xpos[target_object], offset=offset)

        similarity = compute_similarity(move_direction, goal_direction)

        return similarity >= threshold
    
    @staticmethod
    def is_reached(xpos, source_object, target_object, threshold, offset=None):
        distance = compute_distance(xpos[source_object], xpos[target_object], offset=offset)

        return distance <= threshold
    
    @staticmethod
    def is_grasping(xpos, next_xpos):
        closing_finger = xpos["finger_distance"] - next_xpos["finger_distance"] >= 0.0025

        return closing_finger
    
    @staticmethod
    def is_moving(xpos, next_xpos, object_name, threshold):
        move_distance = compute_distance(xpos[object_name], next_xpos[object_name])

        return move_distance >= threshold