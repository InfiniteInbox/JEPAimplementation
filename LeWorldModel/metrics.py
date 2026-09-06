# mainly similar to V-JEPA with a few tweaks

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

@torch.no_grad()
def collect_emb(enc, dataset, device="cuda", batch_size=512):
    # Frozen emb and the actual physical targets associated
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size, shuffle=False)
    enc.eval()
    Z, Y = [], []
    for ctx, _, _, state, physics in loader:
        Z.append(enc(ctx.to(device)).cpu())
        Y.append(torch.cat([state, physics], dim=1))
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()

def probe_r2(Z, Y, targets=('x', 'y', 'vx', 'vy', 'bounce', 'friction'), train_frac=0.8, seed=0, mlp_epochs=200, device="cuda"):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    n_train = int(train_frac * len(Z))
    train, test = idx[:n_train], idx[n_train:]

    mu, sd = Z[train].mean(0), Z[train].std(0)
    Ztrain, Ztest = (Z[train] - mu)/sd , (Z[test] - mu)/sd * 1e-8 # divide be zero AAAAAAAAA
    res = {}
    for j, name in enumerate(targets):
        y_train, y_test = Y[train, j], Y[test, j]

        best_lin = -np.inf
        for a in (.001, .1, 1):
            r = Ridge(alpha=a).fit(Ztrain, y_train)
            best_lin = max(best_lin, r2_score(y_test, r.predict(Ztest)))
        torch.manual_seed(seed)
        mlp = nn.Sequential(nn.Linear(Z.shape[1], 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=.001)
        xtrain = torch.from_numpy(Ztrain).float().to(device)
        ytrain = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
        for _ in range(mlp_epochs):
            opt.zero_grad()
            F.mse_loss(mlp(xtrain), ytrain).backward()
            opt.step()
        with torch.no_grad():
            pred = mlp(torch.from_numpy(Ztest).float().to(device)).cpu().numpy().squeeze()
        mlp_r2 = r2_score(y_test, pred)

        res[name] = {"linear": round(best_lin, 4), "mlp": round(mlp_r2, 4)}
    return res

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
def moment_check(Z):
    #distance from N(0 , I)
    return {"mean_norm": Z.mean(0).norm().item(),
            "std_mean": Z.std(0).mean().item(),
            "std_min": Z.std(0).min().item(),
            "std_max": Z.std(0).max().item()}
@torch.no_grad()
def action_sens(pred, z, n_actions=25):
    #Spread of pred across all actions 
    outs = torch.stack([pred(z, torch.full((z.size(0),), a, dtype=torch.long, device=z.device))
                        for a in range(n_actions)])         
    return outs.std(0).mean().item()

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