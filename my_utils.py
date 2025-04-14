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

    if offset:
        offset = np.array(offset)
        pos_to += offset

    if plane_index:
        pos_from = pos_from[plane_index]
        pos_to = pos_to[plane_index]

    distance = np.linalg.norm(pos_from - pos_to, axis=-1)

    return distance

def compute_direction(pos_from, pos_to, offset=None, plane_index=None):
    pos_from = np.array(pos_from)
    pos_to = np.array(pos_to)

    if offset:
        offset = np.array(offset)
        pos_to += offset

    if plane_index:
        pos_from = pos_from[plane_index]
        pos_to = pos_to[plane_index]

    direction = pos_to - pos_from
    direction /= (np.linalg.norm(direction)+1e-8)

    return direction

def compute_similarity(vec_1, vec_2):
    return np.dot(vec_1, vec_2)

def get_xpos(obs):
    return {
            "gripper": obs[:3],
            "finger_distance": obs[3],
            "target": obs[4:7],
            "goal": obs[-3:]
        }

class DetectMotion:
    @staticmethod
    def is_aligned(xpos, source_object, target_object, threshold, plane_index=[0, 1]):
        plane_distance = compute_distance(xpos[source_object], xpos[target_object], plane_index=plane_index)
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

        is_xy_aligned = xy_distance <= 0.05
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
    
def preprocess(batch_task, tasks, device, numContrast=100):

    targets = torch.tensor([tasks.index(n) for n in batch_task]).to(device)

    return tasks, targets

def compute_logits(frame_feature, text_feature):
    frame_feature = frame_feature / frame_feature.norm(dim=-1, keepdim=True)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    logits = frame_feature @ text_feature.t() / 0.07

    return logits

def accuray(logits, targets):
    preds = logits.softmax(-1).topk(1)[1][:, 0]
    accuray = (preds == targets).tolist()
    return accuray

# Define the RBF kernel function
def rbf_kernel(x, y, bandwidth=1.0):
    dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dist / (2 * bandwidth))

# Compute the MMD loss
def compute_mmd(source_features, target_features, bandwidth=1.0):
    # Compute the kernel matrices
    Kxx = rbf_kernel(source_features, source_features, bandwidth)
    Kyy = rbf_kernel(target_features, target_features, bandwidth)
    Kxy = rbf_kernel(source_features, target_features, bandwidth)
    
    # Compute MMD
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)