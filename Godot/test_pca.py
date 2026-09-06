import numpy as np, torch, json
from pathlib import Path
import sys
sys.path.append("C:/Users/yjain/Desktop/Personal Work/PersonalImprovement/Yan LeCun JEPA")
from LeWorldModel import models
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
# Need to check if the same approach to the V-JEPA will work for the LeWorldModel as SigReg makes the e-rank 103 instead of 30 ish, and when experimenting in Godot, the PCA fit is only a few percentage points
# Need to probe PCA as its linearly decodable but the first two dims wont cut it. 
CKPT = Path("C:/Users/yjain/Desktop/Personal Work/PersonalImprovement/Yan LeCun JEPA/LeWorldModel/checkpoint")
MANIFEST = CKPT.parent / "episodes" / "manifest.json"

ck = torch.load(str(CKPT / "lewm_best.pt"), map_location="cuda")
cfg = ck["config"]
enc = models.encoder(in_channels=cfg["stack"]).cuda()
enc.load_state_dict(ck["enc"]); enc.eval()
stack, k = cfg["stack"], cfg["k"]
print("checkpoint epoch:", ck["epoch"], "| lambda:", cfg["lambda"])

manifest = json.load(open(MANIFEST))
Z, POS = [], []
with torch.no_grad():
    for entry in manifest[:30]:
        d = np.load(entry["path"])
        frames = torch.from_numpy(d["frames"].astype(np.float32)/255.0).cuda()
        states = d["states"]                      
        T = frames.shape[0]
        for t in range(0, T - stack, 2):
            z = enc(frames[t:t+stack].unsqueeze(0)).cpu().numpy()[0]
            Z.append(z)
            POS.append(states[t+stack-1][:2])     
Z = np.array(Z); POS = np.array(POS)
print("embeddings:", Z.shape)

pca = PCA(n_components=10).fit(Z)
print("\ntop-10 explained variance ratio:")
print(np.round(pca.explained_variance_ratio_, 4))
print("sum of top 10:", round(pca.explained_variance_ratio_[:10].sum(), 3))

ev = pca.explained_variance_ratio_
pca_full = PCA().fit(Z)
p = pca_full.explained_variance_ratio_
erank = np.exp(-(p * np.log(p + 1e-12)).sum())
print("effective rank:", round(erank, 1))

n = len(Z); tr = int(0.8*n)
idx = np.random.default_rng(0).permutation(n)
Ztr, Zte = Z[idx[:tr]], Z[idx[tr:]]
Ptr, Pte = POS[idx[:tr]], POS[idx[tr:]]
for j, name in enumerate(["ball_x", "ball_y"]):
    r = Ridge(alpha=1.0).fit(Ztr, Ptr[:, j])
    print(f"linear R^2 {name}: {r2_score(Pte[:, j], r.predict(Zte)):.3f}")


b = np.load("LeWorldModel/checkpoint/pca_basis.npz")
print(b["var_ratio"])