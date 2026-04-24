"""
PWP v2 vs v3 -- Split CIFAR-10
===============================
Harder benchmark than Permuted MNIST.
Split CIFAR-10: 5 tasks, each a binary classification problem
over a disjoint pair of CIFAR-10 classes.

Task 0: airplane vs automobile  (classes 0, 1)
Task 1: bird vs cat             (classes 2, 3)
Task 2: deer vs dog             (classes 4, 5)
Task 3: frog vs horse           (classes 6, 7)
Task 4: ship vs truck           (classes 8, 9)

Each task uses only its 2 classes. Private output head per domain (2 logits).
Input: 32x32 RGB = 3072 dims flattened.

Why this is harder than Permuted MNIST:
- RGB images, not grayscale
- Semantically distinct visual categories per task
- 3072-dim input vs 784
- Binary classification per task vs 10-way -- capacity ceiling bites harder
  because the model must learn rich visual features in a smaller subspace

Config matches v2/v3 MNIST experiments where possible for direct comparison.
H_TOTAL = 640, k = H_TOTAL // N_TASKS = 128 per domain.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_TOTAL     = 640
N_TASKS     = 5
K           = H_TOTAL // N_TASKS   # 128
N_CLASSES   = 2                    # binary per task
INPUT_DIM   = 3072                 # 32x32x3
EPOCHS      = 10                   # more epochs -- harder task
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42
QR_EVERY    = 1

TASK_CLASSES = [
    (0, 1),   # airplane, automobile
    (2, 3),   # bird, cat
    (4, 5),   # deer, dog
    (6, 7),   # frog, horse
    (8, 9),   # ship, truck
]

# ── Data ──────────────────────────────────────────────────────────────────────

def get_split_loaders(base_train, base_test, class_pair):
    """Return loaders for a binary task over class_pair = (c0, c1)."""
    c0, c1 = class_pair

    def filter_and_remap(dataset):
        targets = np.array(dataset.targets)
        idx = np.where((targets == c0) | (targets == c1))[0]
        subset = Subset(dataset, idx)
        # Remap original labels to 0/1
        class_map = {c0: 0, c1: 1}

        class RemappedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, class_map):
                self.subset    = subset
                self.class_map = class_map
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, i):
                x, y = self.subset[i]
                return x, self.class_map[y]

        return RemappedSubset(subset, class_map)

    train_ds = filter_and_remap(base_train)
    test_ds  = filter_and_remap(base_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── Baseline MLP ──────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    def __init__(self, hidden=H_TOTAL, n_tasks=N_TASKS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, N_CLASSES) for _ in range(n_tasks)])

    def forward(self, x, task_id):
        return self.heads[task_id](self.shared(x))

# ── PWP v2 Linear Partition ───────────────────────────────────────────────────

class V2Block(nn.Module):
    def __init__(self, h_d):
        super().__init__()
        self.fc1  = nn.Linear(INPUT_DIM, h_d)
        self.fc2  = nn.Linear(h_d, h_d)
        self.head = nn.Linear(h_d, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))


class V2MLP(nn.Module):
    def __init__(self, n_domains=N_TASKS, hidden=H_TOTAL):
        super().__init__()
        self.h_d    = hidden // n_domains
        self.blocks = nn.ModuleList([V2Block(self.h_d) for _ in range(n_domains)])

    def forward(self, x, domain_id):
        return self.blocks[domain_id](x)

    def parameters_for_domain(self, d):
        return self.blocks[d].parameters()

# ── PWP v3 Grassmannian ───────────────────────────────────────────────────────

def svd_roundrobin_init(n_domains, h, k, seed=SEED):
    torch.manual_seed(seed)
    W_ref = torch.randn(h, h)
    U, _, _ = torch.linalg.svd(W_ref, full_matrices=True)
    bases = []
    for d in range(n_domains):
        indices = [d + j * n_domains for j in range(k)]
        bases.append(U[:, indices])
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            overlap = (bases[i].T @ bases[j]).abs().max().item()
            assert overlap < 1e-5, f"Orthogonality violated: {i},{j} overlap={overlap:.2e}"
    return bases


class GrassmannianLayer(nn.Module):
    def __init__(self, in_features, out_features, bases):
        super().__init__()
        self.n_domains = len(bases)
        self.k         = bases[0].shape[1]
        self.linear    = nn.Linear(in_features, out_features, bias=True)
        for d, P in enumerate(bases):
            self.register_buffer(f"P_{d}", P.clone().float())
        self._active_domain = None
        self._hook_handle   = None

    def get_P(self, d):
        return getattr(self, f"P_{d}")

    def set_active_domain(self, d):
        self._active_domain = d
        if self._hook_handle is not None:
            self._hook_handle.remove()
        P  = self.get_P(d)
        Pi = P @ P.T

        def grad_hook(grad):
            projected = Pi @ grad
            if grad.shape[1] == Pi.shape[0]:
                projected = projected @ Pi
            return projected

        self._hook_handle = self.linear.weight.register_hook(grad_hook)

    def forward(self, x):
        h    = self.linear(x)
        P    = self.get_P(self._active_domain)
        Pi   = P @ P.T
        return h @ Pi.T

    def reorthogonalize(self):
        P_all = torch.cat([self.get_P(d) for d in range(self.n_domains)], dim=1)
        Q, _  = torch.linalg.qr(P_all)
        for d in range(self.n_domains):
            self.get_P(d).copy_(Q[:, d * self.k : (d + 1) * self.k])


class V3MLP(nn.Module):
    def __init__(self, n_domains=N_TASKS, hidden=H_TOTAL, k=K):
        super().__init__()
        self.n_domains = n_domains
        self.k         = k
        print(f"\nInitializing V3MLP: D={n_domains}, H={hidden}, k={k}")
        bases1 = svd_roundrobin_init(n_domains, hidden, k, seed=SEED)
        bases2 = svd_roundrobin_init(n_domains, hidden, k, seed=SEED + 1)
        self.layer1 = GrassmannianLayer(INPUT_DIM, hidden, bases1)
        self.layer2 = GrassmannianLayer(hidden,    hidden, bases2)
        self.relu   = nn.ReLU()
        self.heads  = nn.ModuleList([nn.Linear(hidden, N_CLASSES) for _ in range(n_domains)])

    def forward(self, x, domain_id):
        self.layer1.set_active_domain(domain_id)
        self.layer2.set_active_domain(domain_id)
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        return self.heads[domain_id](h2)

    def reorthogonalize(self):
        self.layer1.reorthogonalize()
        self.layer2.reorthogonalize()

    def parameters_for_domain(self, d):
        return (list(self.layer1.linear.parameters()) +
                list(self.layer2.linear.parameters()) +
                list(self.heads[d].parameters()))

# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, domain_id, step_counter=None, reortho=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.view(x.size(0), -1).to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, domain_id)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if reortho and step_counter is not None:
            step_counter[0] += 1
            if step_counter[0] % QR_EVERY == 0:
                model.reorthogonalize()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, domain_id):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.view(x.size(0), -1).to(DEVICE), y.to(DEVICE)
        correct += (model(x, domain_id).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── Runners ───────────────────────────────────────────────────────────────────

def run(model_name, model, train_loaders, test_loaders, use_reortho=False):
    print(f"\n── {model_name} {'─'*(50-len(model_name))}")
    acc_matrix  = np.zeros((N_TASKS, N_TASKS))
    step_counter = [0]

    for task_id in range(N_TASKS):
        if hasattr(model, 'parameters_for_domain'):
            params = model.parameters_for_domain(task_id)
        else:
            params = model.parameters()

        optimizer = optim.Adam(params, lr=LR)
        print(f"\n  Task {task_id} {TASK_CLASSES[task_id]} ...")

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loaders[task_id], optimizer, task_id,
                               step_counter=step_counter, reortho=use_reortho)
            print(f"    epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

        for prev in range(task_id + 1):
            acc = evaluate(model, test_loaders[prev], prev)
            acc_matrix[task_id][prev] = acc
            print(f"  acc task {prev} after task {task_id}: {acc:.4f}")

    return acc_matrix


def print_summary(results):
    print("\n\n══ RESULTS SUMMARY ══════════════════════════════════════════")
    header = f"{'Task':<6}" + "".join(f"{n:<18}" for n in results.keys())
    print(header)
    print("─" * (6 + 18 * len(results)))

    names = list(results.keys())
    for task_id in range(N_TASKS):
        row = f"  {task_id:<4}"
        for name in names:
            row += f"{results[name][N_TASKS-1][task_id]:<18.4f}"
        print(row)

    print()
    for name, mat in results.items():
        mean = np.mean([mat[N_TASKS-1][i] for i in range(N_TASKS)])
        print(f"  {name} mean: {mean:.4f}")

    if "v2" in results and "v3" in results:
        delta = (np.mean([results["v3"][N_TASKS-1][i] for i in range(N_TASKS)]) -
                 np.mean([results["v2"][N_TASKS-1][i] for i in range(N_TASKS)]))
        print(f"  v3 - v2 delta: {delta:+.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    base_train = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
    base_test  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loaders = []
    test_loaders  = []
    for pair in TASK_CLASSES:
        tr, te = get_split_loaders(base_train, base_test, pair)
        train_loaders.append(tr)
        test_loaders.append(te)

    results = {}

    # Baseline
    baseline = BaselineMLP().to(DEVICE)
    results["baseline"] = run("Baseline MLP", baseline, train_loaders, test_loaders)

    # v2 linear partition
    v2 = V2MLP().to(DEVICE)
    results["v2"] = run("PWP v2 Linear", v2, train_loaders, test_loaders)

    # v3 Grassmannian
    v3 = V3MLP().to(DEVICE)
    results["v3"] = run("PWP v3 Grassmannian", v3, train_loaders, test_loaders, use_reortho=True)

    print_summary(results)

    for name, mat in results.items():
        np.save(f"cifar_{name}_acc_matrix.npy", mat)
    print("\nMatrices saved: cifar_{baseline,v2,v3}_acc_matrix.npy")


if __name__ == "__main__":
    main()