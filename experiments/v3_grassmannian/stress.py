"""
PWP Stress Test -- Domain Scaling Sweep
========================================
Goal: find where v2 breaks under increasing D while v3 holds.

Method:
- Fix H_TOTAL = 640
- Sweep D in [5, 10, 20, 40]
- At each D: H_D = H_TOTAL // D (v2 row slice per domain)
             k   = H_TOTAL // D (v3 subspace dim per domain, same budget)
- Run Permuted MNIST with D tasks (fast, no image processing PTSD)
- Metric: mean final accuracy across all D tasks after sequential training

Expected:
- v2 degrades monotonically as H_D shrinks (fewer neurons = lower ceiling)
- v3 degrades more slowly (domains use directions through full H, not shrinking slices)
- Crossover point = where Grassmannian advantage becomes empirically real

At D=40: H_D = 16 neurons per domain for v2. That should crack it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_TOTAL     = 640
D_SWEEP     = [5, 10, 20, 40]
N_CLASSES   = 10
INPUT_DIM   = 784
EPOCHS      = 5
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42
QR_EVERY    = 1

# ── Permuted MNIST ────────────────────────────────────────────────────────────

class PermutedMNIST(Dataset):
    def __init__(self, base_dataset, permutation):
        self.base = base_dataset
        self.perm = permutation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img.view(-1)[self.perm], label


def make_permutations(n_tasks, seed=SEED):
    rng = np.random.RandomState(seed)
    perms = [torch.arange(INPUT_DIM)]
    for _ in range(n_tasks - 1):
        perms.append(torch.from_numpy(rng.permutation(INPUT_DIM)).long())
    return perms


def get_loaders(base_train, base_test, perm):
    train_ds = PermutedMNIST(base_train, perm)
    test_ds  = PermutedMNIST(base_test,  perm)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True))

# ── v2 Linear Partition ───────────────────────────────────────────────────────

class V2Block(nn.Module):
    def __init__(self, h_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, h_d), nn.ReLU(),
            nn.Linear(h_d, h_d),       nn.ReLU(),
            nn.Linear(h_d, N_CLASSES),
        )
    def forward(self, x):
        return self.net(x)

class V2MLP(nn.Module):
    def __init__(self, n_domains):
        super().__init__()
        assert H_TOTAL % n_domains == 0
        h_d = H_TOTAL // n_domains
        self.blocks = nn.ModuleList([V2Block(h_d) for _ in range(n_domains)])

    def forward(self, x, d):
        return self.blocks[d](x)

    def parameters_for_domain(self, d):
        return self.blocks[d].parameters()

# ── v3 Grassmannian ───────────────────────────────────────────────────────────

def svd_roundrobin_init(n_domains, h, k, seed=SEED):
    torch.manual_seed(seed)
    U, _, _ = torch.linalg.svd(torch.randn(h, h), full_matrices=True)
    bases = []
    for d in range(n_domains):
        idx = [d + j * n_domains for j in range(k)]
        bases.append(U[:, idx])
    return bases


class GrassLayer(nn.Module):
    def __init__(self, in_f, out_f, bases):
        super().__init__()
        self.n  = len(bases)
        self.k  = bases[0].shape[1]
        self.fc = nn.Linear(in_f, out_f, bias=True)
        for d, P in enumerate(bases):
            self.register_buffer(f"P_{d}", P.clone().float())
        self._d    = None
        self._hook = None

    def get_P(self, d): return getattr(self, f"P_{d}")

    def set_domain(self, d):
        self._d = d
        if self._hook: self._hook.remove()
        Pi = self.get_P(d) @ self.get_P(d).T
        def hook(g):
            out = Pi @ g
            if g.shape[1] == Pi.shape[0]: out = out @ Pi
            return out
        self._hook = self.fc.weight.register_hook(hook)

    def forward(self, x):
        h  = self.fc(x)
        Pi = self.get_P(self._d) @ self.get_P(self._d).T
        return h @ Pi.T

    def reortho(self):
        P_all = torch.cat([self.get_P(d) for d in range(self.n)], dim=1)
        Q, _  = torch.linalg.qr(P_all)
        for d in range(self.n):
            self.get_P(d).copy_(Q[:, d*self.k:(d+1)*self.k])


class V3MLP(nn.Module):
    def __init__(self, n_domains):
        super().__init__()
        assert H_TOTAL % n_domains == 0
        k = H_TOTAL // n_domains
        self.k = k
        self.n = n_domains
        b1 = svd_roundrobin_init(n_domains, H_TOTAL, k, seed=SEED)
        b2 = svd_roundrobin_init(n_domains, H_TOTAL, k, seed=SEED+1)
        self.l1    = GrassLayer(INPUT_DIM, H_TOTAL, b1)
        self.l2    = GrassLayer(H_TOTAL,   H_TOTAL, b2)
        self.relu  = nn.ReLU()
        self.heads = nn.ModuleList([nn.Linear(H_TOTAL, N_CLASSES) for _ in range(n_domains)])

    def forward(self, x, d):
        self.l1.set_domain(d); self.l2.set_domain(d)
        return self.heads[d](self.relu(self.l2(self.relu(self.l1(x)))))

    def reortho(self):
        self.l1.reortho(); self.l2.reortho()

    def parameters_for_domain(self, d):
        return list(self.l1.fc.parameters()) + list(self.l2.fc.parameters()) + list(self.heads[d].parameters())

# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, d, step_ctr=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x, d), y)
        loss.backward()
        opt.step()
        if step_ctr is not None:
            step_ctr[0] += 1
            if step_ctr[0] % QR_EVERY == 0 and hasattr(model, 'reortho'):
                model.reortho()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, d):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x, d).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def run_model(model, name, n_domains, train_loaders, test_loaders):
    use_reortho = hasattr(model, 'reortho')
    step_ctr = [0]
    final_accs = []

    for task_id in range(n_domains):
        opt = optim.Adam(model.parameters_for_domain(task_id), lr=LR)
        for _ in range(EPOCHS):
            train_epoch(model, train_loaders[task_id], opt, task_id,
                        step_ctr=step_ctr if use_reortho else None)

    # Evaluate all tasks after all training done
    for task_id in range(n_domains):
        acc = evaluate(model, test_loaders[task_id], task_id)
        final_accs.append(acc)

    mean_acc = np.mean(final_accs)
    min_acc  = np.min(final_accs)
    print(f"    {name:<6}  mean={mean_acc:.4f}  min={min_acc:.4f}  "
          f"per_task={[f'{a:.3f}' for a in final_accs]}")
    return mean_acc, min_acc, final_accs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    base_test  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    summary = []  # (D, h_d, v2_mean, v3_mean)

    for D in D_SWEEP:
        h_d = H_TOTAL // D
        k   = h_d
        print(f"\n{'='*60}")
        print(f"D={D}  H_D={h_d}  k={k}  (H_TOTAL={H_TOTAL})")
        print(f"{'='*60}")

        perms        = make_permutations(D)
        train_loaders = []
        test_loaders  = []
        for p in perms:
            tr, te = get_loaders(base_train, base_test, p)
            train_loaders.append(tr)
            test_loaders.append(te)

        v2 = V2MLP(D).to(DEVICE)
        v3 = V3MLP(D).to(DEVICE)

        v2_mean, v2_min, _ = run_model(v2, "v2", D, train_loaders, test_loaders)
        v3_mean, v3_min, _ = run_model(v3, "v3", D, train_loaders, test_loaders)

        summary.append((D, h_d, v2_mean, v3_mean, v2_min, v3_min))

    print(f"\n\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'D':<6} {'H_D':<8} {'v2 mean':<14} {'v3 mean':<14} {'delta':>10}")
    print("-" * 54)
    for D, h_d, v2m, v3m, v2min, v3min in summary:
        delta = v3m - v2m
        flag  = " <-- v3 wins" if delta > 0.005 else ""
        print(f"{D:<6} {h_d:<8} {v2m:<14.4f} {v3m:<14.4f} {delta:>+10.4f}{flag}")

    np.save("sweep_summary.npy", np.array(summary))
    print("\nSaved: sweep_summary.npy")


if __name__ == "__main__":
    main()