import numpy as np
import torch
import torch.nn.functional as F
@torch.no_grad()
def effective_rank(Z): #More robust than regular rank by creating a probability dist of singular values and calculating e^shannon entropy to account how varied the data is distributed across dims. 
    Z = Z.double()
    Z = Z - Z.mean(0)
    gram = Z.T @ Z / (Z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(gram).clamp(min=0) #Uses decomposition tricks for complex eiegenvalues and other decomp tricks for faster eigvals
    if ev.sum() < 1e-8:
        return torch.tensor(1.0)
    p = ev / ev.sum()
    p = p.clamp(min=1e-12) #clamp clips all eleents in a tensor to a spec min and max
    return torch.exp(-(p * p.log()).sum())


@torch.no_grad()
def rel_L1(pred, target):
    return (F.l1_loss(pred, target) / target.abs().mean().clamp(min=1e-8)).item()

@torch.no_grad()
def embedding_std(Z):
    return Z.std(0).mean()

@torch.no_grad()
def teacher_student_gap(online, target):
    diff_sq, norm_sq = 0.0, 0.0
    for p_o, p_t in zip(online.parameters(), target.parameters(), strict=True):
        diff_sq += (p_o - p_t).pow(2).sum().item()
        norm_sq += p_o.pow(2).sum().item()
    return (diff_sq ** 0.5) / (norm_sq ** 0.5 + 1e-12) # Euclidean / Normal Vec

@torch.no_grad()
def rollout_error(x_enc, y_enc, pred, frames, t0, steps, stack, k): # New error function for multi frame stack batch
    z = x_enc(frames[t0:t0+stack].unsqueeze(0))
    errs = []
    for n in range(1, steps+1):
        z = pred(z)
        s = t0 + n*k
        zt = y_enc(frames[s:s+stack].unsqueeze(0))
        errs.append(rel_L1(z, zt))
    return errs