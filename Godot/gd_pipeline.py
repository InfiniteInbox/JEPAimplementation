from pathlib import Path
import sys
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
# Have to use Ridge regression for PCA as the variance captured by using the same approach as V-JEPA was too small
# This is because SIGreg's goal is to move everything to an isotropic gaussian, meaning that the variance among dims is spread evenly (ish)
# V-JEPA is anisotopic in nature meaning that the model is free to let the majority of its variance rest on a few dims
HERE = Path(__file__).resolve().parent
OUT  = HERE.parent / "LeWorldModel" / "episodes"
CKPT = HERE.parent / "LeWorldModel" / "checkpoint"
MANIFEST = OUT / "manifest.json"

parent_dir = str(HERE.parent)
sys.path.append(parent_dir)

from LeWorldModel import data
from LeWorldModel import models
from LeWorldModel import SigReg
from LeWorldModel import metrics

BASIS_PATH = CKPT / "pca_basis.npz"
BASIS_SCHEMA = "lewm-basis-2"
VIZ_SCHEMA = "lewm-viz-2"
EPS = 1e-8
_CACHE = {}


def create_batch(dict_data, report=None):
    if report:
        report({"kind": "debug", "out": str(OUT), "cwd": os.getcwd()})
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = []
    N_EPISODES = 1000
    for ep in range(N_EPISODES):
        rng = np.random.default_rng(ep)
        r  = rng.uniform(*dict_data.get("ball_radius", [3, 8]))
        x  = rng.uniform(r, 64 - r)
        y  = rng.uniform(r, 64 - r)
        vx = rng.uniform(*dict_data.get("velocityX", [-5, 5]))
        vy = rng.uniform(*dict_data.get("velocityY", [-5, 5]))
        hold = int(rng.integers(*dict_data.get("hold", [2, 9])))
        bounce   = rng.uniform(*dict_data.get("bounce", [0.5, 1.0]))
        friction = rng.uniform(*dict_data.get("friction", [0.0, 0.5]))

        frames, states, actions = data.physics_process(
            x, y, vx, vy, rng, r=r, bounce=bounce,
            friction=friction, hold=hold, frames=120, dt=0.05)

        path = str(OUT / f"ep_{ep:04d}.npz")
        np.savez_compressed(path, frames=frames, states=states, actions=actions,
                            radius=r, bounce=bounce, friction=friction,
                            hold=hold, dt=0.05, seed=ep)
        manifest.append({"path": path, "radius": float(r), "bounce": float(bounce),
                         "friction": float(friction), "hold": hold, "seed": ep})

        if report and (ep % 10 == 0 or ep == N_EPISODES - 1):
            report({"kind": "progress", "done": ep + 1, "total": N_EPISODES})

    json.dump(manifest, open(MANIFEST, "w"), indent=2)


class BounceDataset(Dataset):
    def __init__(self, manifest_path, episode_ids, stack, k, n_roll=4):
        self.stack, self.k, self.n_roll = stack, k, n_roll
        manifest = json.load(open(manifest_path))

        self.frames, self.states, self.actions, self.physics = [], [], [], []
        self.index = []
        for entry in manifest:
            if entry["seed"] not in episode_ids:
                continue
            d = np.load(entry["path"])
            self.frames.append(d["frames"])
            self.states.append(d["states"])
            self.actions.append(d["actions"])
            self.physics.append(np.array([entry["bounce"], entry["friction"]], dtype=np.float32))

            slot = len(self.frames) - 1
            T = d["frames"].shape[0]
            for t in range(T - stack - n_roll * k + 1):
                self.index.append((slot, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        slot, t = self.index[i]
        f, s, a = self.frames[slot], self.states[slot], self.actions[slot]

        ctx = f[t : t + self.stack]
        tgts = np.stack([f[t + n*self.k : t + n*self.k + self.stack]
                         for n in range(1, self.n_roll + 1)])
        raw = a[t + self.stack - 1 : t + self.stack - 1 + self.n_roll]
        acts = (raw[:, 0] + 2) * 5 + (raw[:, 1] + 2)

        ctx  = torch.from_numpy(ctx.astype(np.float32) / 255.0)
        tgts = torch.from_numpy(tgts.astype(np.float32) / 255.0)
        acts = torch.from_numpy(acts.astype(np.int64))
        state = torch.from_numpy(s[t + self.stack - 1])
        physics = torch.from_numpy(self.physics[slot])
        return ctx, tgts, acts, state, physics


def train(report=None, STACK=8, K_STRIDE=2, N_ROLL=4, EPOCHS=80, BS=256,
          LR=1e-4, WD=0.05, LAMBDA=0.1, N_ACTIONS=25, DEV="cuda"):

    CKPT.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(MANIFEST))
    seeds = np.array([e["seed"] for e in manifest])
    np.random.default_rng(0).shuffle(seeds)
    n_val = int(0.15 * len(seeds))
    val_id   = set(seeds[:n_val].tolist())
    train_id = set(seeds[n_val:].tolist())

    train_ds = BounceDataset(str(MANIFEST), train_id, STACK, K_STRIDE, n_roll=N_ROLL)
    val_ds   = BounceDataset(str(MANIFEST), val_id,   STACK, K_STRIDE, n_roll=N_ROLL)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BS, shuffle=False)

    m_ctx, m_tgts, m_acts, m_state, m_phys = next(iter(val_loader))
    m_ctx, m_tgts, m_acts = m_ctx.to(DEV), m_tgts.to(DEV), m_acts.to(DEV)

    enc = models.encoder(in_channels=STACK).to(DEV)
    pred_net = models.predictor(latent=128, n_actions=N_ACTIONS).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred_net.parameters()),
                            lr=LR, weight_decay=WD)

    a_check = next(iter(train_loader))[2]
    if report is not None:
        report({"kind": "preflight", "n_actions": pred_net.action_embedding.num_embeddings,
          "act_min": int(a_check.min()), "act_max": int(a_check.max())})

    step = 0
    history, best_score = [], float("inf")

    for epoch in range(EPOCHS):
        enc.train(); pred_net.train()
        for ctx, tgts, acts, _, _ in train_loader:
            ctx, tgts, acts = ctx.to(DEV), tgts.to(DEV), acts.to(DEV)

            z = enc(ctx)
            z_ctx, z_tgts = z, []
            loss_pred = 0.0
            for n in range(N_ROLL):
                z = pred_net(z, acts[:, n])
                zt = enc(tgts[:, n])
                z_tgts.append(zt)
                loss_pred = loss_pred + F.l1_loss(z, zt)
            loss_pred = loss_pred / N_ROLL

            loss_sig = SigReg.sigreg(torch.cat([z_ctx] + z_tgts), step)
            loss = loss_pred + LAMBDA * loss_sig

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred_net.parameters()), 1.0)
            opt.step()
            step += 1

        enc.eval(); pred_net.eval()
        with torch.no_grad():
            z0 = enc(m_ctx)
            z = z0
            roll, ident = [], []
            for n in range(N_ROLL):
                z = pred_net(z, m_acts[:, n])
                zt = enc(m_tgts[:, n])
                roll.append(metrics.rel_L1(z, zt))
                ident.append(metrics.rel_L1(z0, zt))
            mom = metrics.moment_check(z0)
            erank = float(metrics.effective_rank(z0))
            top = z0.mean(0).abs().topk(10)

        roll_score = float(sum(roll))
        if report is not None:
            report({
                "kind": "train",
                "epoch": epoch, "total_epochs": EPOCHS,
                "pred": round(loss_pred.item(), 4),
                "sig": round(loss_sig.item(), 4),
                "roll":  [round(float(r), 4) for r in roll],
                "ident": [round(float(i), 4) for i in ident],
                "erank": round(erank, 2),
                "mean_norm": round(mom["mean_norm"], 3),
                "std_mean": round(mom["std_mean"], 3),
                "std_min":  round(mom["std_min"], 3),
                "std_max":  round(mom["std_max"], 3),
                "top_dims": top.indices.tolist()   
            })

        history.append({"epoch": epoch, "pred": loss_pred.item(), "sig": loss_sig.item(),
                        "erank": erank, "mean_norm": mom["mean_norm"],
                        "roll": [float(r) for r in roll], "roll_score": roll_score})

        if roll_score < best_score:
            best_score = roll_score
            torch.save({
                "enc": enc.state_dict(),
                "pred": pred_net.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch, "step": step, "history": history,
                "config": {
                    "stack": STACK, "k": K_STRIDE, "n_roll": N_ROLL,
                    "latent": 128, "n_actions": N_ACTIONS, "n_dim": 48, "hidden": 1024,
                    "lambda": LAMBDA, "lr": LR, "wd": WD,
                    "regularizer": "sigreg_ep",
                    "sigreg": {"num_slices": 256, "t_points": 25, "t_range": [-5, 5]},
                },
            }, str(CKPT / "lewm_best.pt"))

        if epoch in (0, 5):
            with torch.no_grad():
                if report is not None:
                    report({"kind": "train_extra", "epoch": epoch,
                      "action_sensitivity": round(float(metrics.action_sens(pred_net, enc(m_ctx))), 4)})

        if epoch % 10 == 0:
            Z, Y = metrics.collect_emb(enc, val_ds)
            probe = {k: round(v["linear"], 4) for k, v in metrics.probe_r2(Z, Y).items()}
            if report is not None:
                report({"kind": "train_probe", "epoch": epoch, "linear_r2": probe})
    if report is not None:
        report({"kind": "train_done", "epochs": EPOCHS, "best_score": round(best_score, 4)})


@torch.no_grad()
def _collect_Z_POS(enc, stack, k, n_ep=40, device="cuda"):
    manifest = json.load(open(MANIFEST))
    Z, POS = [], []
    for entry in manifest[:n_ep]:
        d = np.load(entry["path"])
        frames = torch.from_numpy(d["frames"].astype(np.float32)/255.0).to(device)
        states = d["states"]
        for t in range(0, frames.shape[0] - stack, 2):
            Z.append(enc(frames[t:t+stack].unsqueeze(0)).cpu().numpy()[0])
            POS.append(states[t+stack-1][:2])
    return np.array(Z), np.array(POS)


def build_pca_basis(report=None, device="cuda"):
    from sklearn.linear_model import Ridge

    ck = torch.load(str(CKPT / "lewm_best.pt"), map_location=device)
    cfg = ck["config"]
    enc = models.encoder(in_channels=cfg["stack"]).to(device)
    enc.load_state_dict(ck["enc"]); enc.eval()

    Z, POS = _collect_Z_POS(enc, cfg["stack"], cfg["k"], device=device)

    mean = Z.mean(0)
    Zc = Z - mean

    rx = Ridge(alpha=1.0).fit(Zc, POS[:, 0])
    ry = Ridge(alpha=1.0).fit(Zc, POS[:, 1])

    r2x = rx.score(Zc, POS[:, 0])
    r2y = ry.score(Zc, POS[:, 1])

    # calibrated predictions, in real ball-position units, replace the raw projection
    pred_x = rx.predict(Zc)
    pred_y = ry.predict(Zc)
    proj = np.stack([pred_x, pred_y], axis=1)     # [N, 2], units = ball pixels

    view_lo = proj.min(axis=0)
    view_hi = proj.max(axis=0)
    latent_scale = float(np.linalg.norm(Zc, axis=1).mean())

    np.savez(str(BASIS_PATH),
             coef=np.stack([rx.coef_, ry.coef_]).astype(np.float32),      # [2, 128]
             intercept=np.array([rx.intercept_, ry.intercept_], dtype=np.float32),  # [2]
             mean=mean.astype(np.float32),
             var_ratio=np.array([r2x, r2y], dtype=np.float32),
             view_lo=view_lo.astype(np.float32),
             view_hi=view_hi.astype(np.float32),
             latent_scale=np.float32(latent_scale))
    _CACHE.clear()
    if report:
        report({"kind": "extract_done", "var_ratio": [round(float(r2x), 3), round(float(r2y), 3)]})

def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def _basis_is_current() -> bool:
    if not BASIS_PATH.exists():
        return False
    try:
        d = np.load(str(BASIS_PATH), allow_pickle=False)
    except Exception:
        return False
    return "schema" in d and str(d["schema"]) == BASIS_SCHEMA


def _load_once(device="cuda"):
    if not _CACHE:
        if not _basis_is_current():
            build_pca_basis(device=device)
        ck = torch.load(str(CKPT / "lewm_best.pt"), map_location=device)
        cfg = ck["config"]
        enc = models.encoder(in_channels=cfg["stack"]).to(device)
        pred = models.predictor(latent=cfg["latent"], n_actions=cfg["n_actions"]).to(device)
        enc.load_state_dict(ck["enc"]); pred.load_state_dict(ck["pred"])
        enc.eval(); pred.eval()
        b = np.load(str(BASIS_PATH))
        _CACHE.update(
            enc=enc, pred=pred, stack=cfg["stack"], k=cfg["k"],
            coef=b["coef"].astype(np.float64),
            intercept=b["intercept"].astype(np.float64),
            mean=b["mean"].astype(np.float64),
            var=b["var_ratio"].astype(np.float64),
            latent_scale=float(b["latent_scale"]),
            view_lo=b["view_lo"].astype(np.float64),
            view_hi=b["view_hi"].astype(np.float64),
        )
    return _CACHE


def _fget(d, *keys, default=0.0):
    for key in keys:
        if key in d and d[key] is not None:
            return float(d[key])
    return float(default)


def _act_id(a) -> int:
    # matches BounceDataset: (ax + 2) * 5 + (ay + 2), ax/ay in {-2..2}
    return int((int(round(float(a[0]))) + 2) * 5 + (int(round(float(a[1]))) + 2))


# --------------------------------------------------------------------------- #
# visualization
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_visualization(dict_data, report=None, n_steps=20, device="cuda"):
    c = _load_once(device)
    enc, pred, stack, k = c["enc"], c["pred"], c["stack"], c["k"]

    rng = np.random.default_rng(int(dict_data.get("seed", 0)))
    x = _fget(dict_data, "x", default=32)
    y = _fget(dict_data, "y", default=32)
    vx = _fget(dict_data, "vx", "velocityX", default=0)
    vy = _fget(dict_data, "vy", "velocityY", default=0)
    r = _fget(dict_data, "ball_radius", "r", default=5)
    bounce = _fget(dict_data, "bounce", default=0.9)
    friction = _fget(dict_data, "friction", default=0.2)
    hold = max(1, int(round(float(dict_data.get("hold", 4)))))

    frames_np, states, actions = data.physics_process(
        x, y, vx, vy, rng,
        r=r, bounce=bounce, friction=friction, hold=hold,
        frames=120, dt=0.05)
    frames = torch.from_numpy(frames_np.astype(np.float32) / 255.0).to(device)
    states = np.asarray(states)
    actions = np.asarray(actions)

    last = frames.shape[0] - stack
    idxs = [min(n * k, last) for n in range(n_steps + 1)]

    # One batched forward instead of n_steps + 1 single-sample calls. This is
    # what makes the panel keep up with a slider being dragged.
    windows = torch.stack([frames[s:s + stack] for s in idxs])
    true = enc(windows).cpu().numpy().astype(np.float64)

    z = enc(frames[0:stack].unsqueeze(0))
    imag = [z]
    for n in range(n_steps):
        t_act = min(stack - 1 + n, len(actions) - 1)
        z = pred(z, torch.tensor([_act_id(actions[t_act])], device=device))
        imag.append(z)
    imag = torch.cat(imag).cpu().numpy().astype(np.float64)

    comp, mean, scale = c["coef"], c["mean"], max(c["latent_scale"], EPS)
    proj = lambda Z: (Z - mean) @ comp.T + c["intercept"]     # calibrated prediction, not raw projection
    T, I = proj(true), proj(imag)

    # Contraction: the imagined step over the step the world actually took.
    # The old ratio divided by the model's own first step, which cancels any
    # constant under-step exactly and leaves you plotting the ball's speed.
    n_true = np.linalg.norm(np.diff(true, axis=0), axis=1)
    n_imag = np.linalg.norm(np.diff(imag, axis=0), axis=1)
    step_ratio_true = n_imag / np.maximum(n_true, EPS)

    # Surprise, in units of the basis latent scale so the panel axis can be a
    # constant and two payloads are comparable.
    surprise = np.linalg.norm(imag - true, axis=1) / scale
    surprise_ident = np.linalg.norm(true - true[0:1], axis=1) / scale

    t_idx = [min(s + stack - 1, len(states) - 1) for s in idxs]
    if states.ndim == 2 and states.shape[1] > 1:
        heights = states[t_idx, 1].astype(np.float32)
    else:
        heights = np.zeros(len(idxs), dtype=np.float32)

    std = true.std(axis=0)

    payload = {
        "kind": "viz",
        "schema": VIZ_SCHEMA,
        "seq": int(dict_data.get("seq", 0)),
        "true": T.astype(np.float32).tolist(),
        "imagined": I.astype(np.float32).tolist(),
        "heights": [float(h) for h in heights],
        "var_ratio": [float(v) for v in c["var"][:2]],
        "view_lo": [float(v) for v in c["view_lo"]],
        "view_hi": [float(v) for v in c["view_hi"]],
        "latent_scale": float(scale),
        "step_ratio_true": [float(v) for v in step_ratio_true],
        "surprise": [float(v) for v in surprise],
        "surprise_ident": [float(v) for v in surprise_ident],
        "mean_norm": round(float(np.linalg.norm(true.mean(axis=0) - mean) / scale), 4),
        "std_mean": round(float(std.mean()), 4),
        "std_min": round(float(std.min()), 4),
        "std_max": round(float(std.max()), 4),
        "params": {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "r": r, "bounce": bounce, "friction": friction, "hold": hold,
        },
    }
    if report:
        report(payload)
    return payload
