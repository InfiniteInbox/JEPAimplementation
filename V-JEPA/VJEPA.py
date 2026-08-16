import torch 
import torch.nn as nn
import copy
import models

@torch.no_grad()
def exponential_moving_average(online, target, m):
    #Extract paired params
    for online_p, target_p in zip(online.parameters(), target.parameters(), strict=True):
        target_p.lerp_(online_p, 1-m) # θˉ + (1-m)(θ - θˉ)
    for online_b, target_b in zip(online.buffers(), target.buffers()):
        target_b.copy_(online_b)

    

