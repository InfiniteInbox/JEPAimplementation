# An implementation of Sigreg from the LeJEPA paper
# TODO: Maybe try VISReg (Wu et al., June 2026) later?
import torch
def sigreg(x, global_step, num_slices=256):
    # create a torch generator on the x.device, manual seed, then Torch randn (K, num_slices) since x is [N, K]
    # Normalize all cols 
    N, K = x.shape
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)

    A = torch.randn((K, num_slices), generator=g, device=x.device, dtype=x.dtype)
    A = A / A.norm(p=2, dim=0)

    # In the Le-JEPA paper, they state that for the trapezoidal approx of the integral, 25 evenly spaced points is very effective
    t = torch.linspace(-5,5,25,device=x.device) 
    exp_f = torch.exp(-0.5 * t**2)

    # projections
    proj = x @ A # Matmul to compute all 1d directions of the x matrix 
    x_t = proj.unsqueeze(2) * t # Sample at each integration point from t
    ecf = (1j * x_t).exp().mean(0) # Emperical Charecteristic Function (Fourier trasform of the Emperical measue)
    err = (ecf - exp_f).abs().square().mul(exp_f)    #actual abs squared gap:  [M, 17], real

    T = torch.trapz(err, t, dim=-1) * x.size(0) # Performs the trapezoidal approx [M] for each dim 
    return T.mean()


# LLM Test Suite
'''
print(sigreg(torch.randn(256, 128, device="cuda"), 0))        # within small range
print(sigreg(torch.zeros(256, 128, device="cuda") + 0.01, 0))  # large 
print(sigreg(torch.randn(256, 128, device="cuda") + 3.0, 0))   # large 
print(sigreg(torch.randn(256, 128, device="cuda") * 0.3, 0))  # somewhat large
for s in range(5):
    print(sigreg(torch.randn(256, 128, device="cuda"), s).item()) #within small range

x = torch.cat([torch.randn(128,128) - 2, torch.randn(128, 128) + 2]).cuda().requires_grad_()
opt = torch.optim.Adam([x], lr=0.05)

dim_idx = 0
x_before = x[:, dim_idx].detach().cpu().numpy()

for step in range(500):
    loss = sigreg(x, step)
    opt.zero_grad()
    loss.backward()
    opt.step()
x_after = x[:, dim_idx].detach().cpu().numpy()

print(f"Before - Mean: {x_before.mean():.3f}, Std: {x_before.std():.3f}")
print(f"After  - Mean: {x_after.mean():.3f}, Std: {x_after.std():.3f}")
'''