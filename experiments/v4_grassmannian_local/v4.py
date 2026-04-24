"""
PWP v4 -- Domain-Local Orthogonality: Permuted MNIST Sweep
===========================================================
Fix for v3's QR drift problem at high D.

v3 problem:
    Global QR re-orthogonalization sweeps ALL D basis matrices together
    every step. Early trained subspaces get nudged by every subsequent
    domain's training. At D=40 with k=16, this accumulates into
    measurable retention loss on early tasks.

v4 fix:
    When domain d begins training:
        1. Project P_d into the null space of span{P_0, ..., P_{d-1}}
           via incremental Gram-Schmidt against frozen bases only.
        2. P_d now lives in a subspace orthogonal to all prior domains
           by construction, before any gradient step.
    During domain d's training:
        3. Gradient hook gates to P_d's subspace (same as v3).
        4. P_d is free to rotate within its orthogonal complement.
    After domain d's training completes:
        5. Freeze P_d. It is never modified again.

    No global QR. No touching frozen bases. Early domains are
    structurally immutable after their training window closes.

Same sweep as stress.py: D in [5, 10, 20, 40], H_TOTAL=640.
Direct comparison: baseline, v2, v3, v4.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_TOTAL   = 640
D_SWEEP   = [5, 10, 20, 40]
N_CLASSES = 10
INPUT_DIM = 784
EPOCHS    = 5
BATCH_SIZE= 256
LR        = 1e-3
SEED      = 42

# ── Permuted MNIST ────────────────────────────────────────────────────────────

class PermutedMNIST(Dataset):
    def __init__(self, base, perm):
        self.base = base
        self.perm = perm
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x.view(-1)[self.perm], y

def make_permutations(n, seed=SEED):
    rng = np.random.RandomState(seed)
    perms = [torch.arange(INPUT_DIM)]
    for _ in range(n - 1):
        perms.append(torch.from_numpy(rng.permutation(INPUT_DIM)).long())
    return perms

def get_loaders(base_train, base_test, perm):
    tr = DataLoader(PermutedMNIST(base_train, perm), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    te = DataLoader(PermutedMNIST(base_test,  perm), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return tr, te

# ── v2 Linear Partition ───────────────────────────────────────────────────────

class V2Block(nn.Module):
    def __init__(self, h_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, h_d), nn.ReLU(),
            nn.Linear(h_d, h_d),       nn.ReLU(),
            nn.Linear(h_d, N_CLASSES),
        )
    def forward(self, x): return self.net(x)

class V2MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        assert H_TOTAL % n == 0
        self.blocks = nn.ModuleList([V2Block(H_TOTAL // n) for _ in range(n)])
    def forward(self, x, d): return self.blocks[d](x)
    def parameters_for_domain(self, d): return self.blocks[d].parameters()

# ── Shared Grassmannian Layer ─────────────────────────────────────────────────

def random_orthonormal(h, k, seed):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(h, k))
    return Q  # H x k, orthonormal columns

def incremental_gs(P_new, frozen_bases):
    """
    Project P_new (H x k) into the null space of all frozen_bases.
    For each frozen basis P_f, remove P_f's component from P_new column by column.
    Then re-orthonormalize P_new internally via QR.
    Returns the projected, orthonormal P_new.
    """
    V = P_new.clone()
    for P_f in frozen_bases:
        # Remove frozen subspace components: V -= P_f (P_f^T V)
        V = V - P_f @ (P_f.T @ V)
    # Re-orthonormalize what remains
    Q, R = torch.linalg.qr(V)
    # Check for rank collapse -- if any column near-zero, subspace is exhausted
    diag = R.diag().abs()
    if (diag < 1e-6).any():
        print(f"    WARNING: subspace near-collapse detected. "
              f"min diagonal={diag.min():.2e}. "
              f"Consider reducing D or increasing H_TOTAL.")
    return Q  # H x k, orthonormal, orthogonal to all frozen bases


class V4Layer(nn.Module):
    """
    Grassmannian layer with domain-local orthogonality enforcement.
    Frozen bases are registered once and never modified.
    Active domain's basis is projected into null space of frozen bases
    before training begins, then frozen after training completes.
    """
    def __init__(self, in_f, out_f, n_domains, k, seed_offset=0):
        super().__init__()
        self.n   = n_domains
        self.k   = k
        self.fc  = nn.Linear(in_f, out_f, bias=True)

        # Initialize all bases randomly orthonormal
        for d in range(n_domains):
            P = random_orthonormal(out_f, k, seed=SEED + seed_offset + d)
            self.register_buffer(f"P_{d}", P.float())

        self.frozen     = set()   # domain ids that have been trained and locked
        self._active    = None
        self._hook      = None

    def get_P(self, d): return getattr(self, f"P_{d}")

    def prepare_domain(self, d):
        """
        Call before training domain d.
        Projects P_d into null space of all frozen domains.
        """
        frozen_bases = [self.get_P(f).detach() for f in sorted(self.frozen)]
        if frozen_bases:
            P_new = incremental_gs(self.get_P(d).detach(), frozen_bases)
            self.get_P(d).copy_(P_new)

    def freeze_domain(self, d):
        """Call after training domain d completes. Locks the basis permanently."""
        self.frozen.add(d)

    def set_active(self, d):
        self._active = d
        if self._hook: self._hook.remove()
        Pi = self.get_P(d) @ self.get_P(d).T
        def hook(g):
            out = Pi @ g
            if g.shape[1] == Pi.shape[0]: out = out @ Pi
            return out
        self._hook = self.fc.weight.register_hook(hook)

    def forward(self, x):
        h  = self.fc(x)
        Pi = self.get_P(self._active) @ self.get_P(self._active).T
        return h @ Pi.T


class V4MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        assert H_TOTAL % n == 0
        k = H_TOTAL // n
        self.k = k
        self.n = n
        self.l1    = V4Layer(INPUT_DIM, H_TOTAL, n, k, seed_offset=0)
        self.l2    = V4Layer(H_TOTAL,   H_TOTAL, n, k, seed_offset=100)
        self.relu  = nn.ReLU()
        self.heads = nn.ModuleList([nn.Linear(H_TOTAL, N_CLASSES) for _ in range(n)])

    def prepare_domain(self, d):
        self.l1.prepare_domain(d)
        self.l2.prepare_domain(d)

    def freeze_domain(self, d):
        self.l1.freeze_domain(d)
        self.l2.freeze_domain(d)

    def forward(self, x, d):
        self.l1.set_active(d)
        self.l2.set_active(d)
        return self.heads[d](self.relu(self.l2(self.relu(self.l1(x)))))

    def parameters_for_domain(self, d):
        return (list(self.l1.fc.parameters()) +
                list(self.l2.fc.parameters()) +
                list(self.heads[d].parameters()))

# ── v3 (global QR, for comparison) ───────────────────────────────────────────

def svd_roundrobin_init(n, h, k, seed=SEED):
    torch.manual_seed(seed)
    U, _, _ = torch.linalg.svd(torch.randn(h, h), full_matrices=True)
    return [U[:, [d + j*n for j in range(k)]].clone().float() for d in range(n)]

class V3Layer(nn.Module):
    def __init__(self, in_f, out_f, bases):
        super().__init__()
        self.n = len(bases); self.k = bases[0].shape[1]
        self.fc = nn.Linear(in_f, out_f, bias=True)
        for d, P in enumerate(bases): self.register_buffer(f"P_{d}", P)
        self._d = None; self._hook = None
    def get_P(self, d): return getattr(self, f"P_{d}")
    def set_active(self, d):
        self._d = d
        if self._hook: self._hook.remove()
        Pi = self.get_P(d) @ self.get_P(d).T
        def hook(g):
            out = Pi @ g
            if g.shape[1] == Pi.shape[0]: out = out @ Pi
            return out
        self._hook = self.fc.weight.register_hook(hook)
    def forward(self, x):
        h = self.fc(x); Pi = self.get_P(self._d) @ self.get_P(self._d).T
        return h @ Pi.T
    def reortho(self):
        P_all = torch.cat([self.get_P(d) for d in range(self.n)], dim=1)
        Q, _ = torch.linalg.qr(P_all)
        for d in range(self.n): self.get_P(d).copy_(Q[:, d*self.k:(d+1)*self.k])

class V3MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        assert H_TOTAL % n == 0
        k = H_TOTAL // n
        self.k = k; self.n = n
        self.l1 = V3Layer(INPUT_DIM, H_TOTAL, svd_roundrobin_init(n, H_TOTAL, k, SEED))
        self.l2 = V3Layer(H_TOTAL,   H_TOTAL, svd_roundrobin_init(n, H_TOTAL, k, SEED+1))
        self.relu = nn.ReLU()
        self.heads = nn.ModuleList([nn.Linear(H_TOTAL, N_CLASSES) for _ in range(n)])
    def forward(self, x, d):
        self.l1.set_active(d); self.l2.set_active(d)
        return self.heads[d](self.relu(self.l2(self.relu(self.l1(x)))))
    def reortho(self): self.l1.reortho(); self.l2.reortho()
    def parameters_for_domain(self, d):
        return list(self.l1.fc.parameters()) + list(self.l2.fc.parameters()) + list(self.heads[d].parameters())

# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_domain(model, loader, d, use_reortho=False):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters_for_domain(d), lr=LR)
    step = [0]
    for _ in range(EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x, d), y).backward()
            opt.step()
            if use_reortho:
                step[0] += 1
                model.reortho()

@torch.no_grad()
def evaluate(model, loader, d):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x, d).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

def run_sequential(model, name, n, train_loaders, test_loaders, use_reortho=False, use_v4=False):
    for task_id in range(n):
        if use_v4:
            model.prepare_domain(task_id)
        train_domain(model, train_loaders[task_id], task_id, use_reortho=use_reortho)
        if use_v4:
            model.freeze_domain(task_id)

    accs = [evaluate(model, test_loaders[i], i) for i in range(n)]
    mean = np.mean(accs)
    mn   = np.min(accs)
    print(f"    {name:<6}  mean={mean:.4f}  min={mn:.4f}")
    return mean, mn

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    base_test  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    summary = []

    for D in D_SWEEP:
        print(f"\n{'='*60}")
        print(f"D={D}  H_D={H_TOTAL//D}  k={H_TOTAL//D}  (H_TOTAL={H_TOTAL})")
        print(f"{'='*60}")

        perms = make_permutations(D)
        train_loaders = []; test_loaders = []
        for p in perms:
            tr, te = get_loaders(base_train, base_test, p)
            train_loaders.append(tr); test_loaders.append(te)

        v2 = V2MLP(D).to(DEVICE)
        v3 = V3MLP(D).to(DEVICE)
        v4 = V4MLP(D).to(DEVICE)

        v2m, _ = run_sequential(v2, "v2", D, train_loaders, test_loaders)
        v3m, _ = run_sequential(v3, "v3", D, train_loaders, test_loaders, use_reortho=True)
        v4m, _ = run_sequential(v4, "v4", D, train_loaders, test_loaders, use_v4=True)

        summary.append((D, H_TOTAL//D, v2m, v3m, v4m))

    print(f"\n\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'D':<6} {'H_D':<8} {'v2':<12} {'v3':<12} {'v4':<12} {'v4-v2':>8} {'v4-v3':>8}")
    print("-" * 58)
    for D, h_d, v2m, v3m, v4m in summary:
        flag = " <--" if v4m > v2m + 0.003 and v4m > v3m else ""
        print(f"{D:<6} {h_d:<8} {v2m:<12.4f} {v3m:<12.4f} {v4m:<12.4f} "
              f"{v4m-v2m:>+8.4f} {v4m-v3m:>+8.4f}{flag}")

    np.save("v4_sweep_summary.npy", np.array(summary))
    print("\nSaved: v4_sweep_summary.npy")


if __name__ == "__main__":
    main()