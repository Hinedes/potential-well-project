"""
PWP v3 -- Grassmannian Subspace Partition: Permuted MNIST
=========================================================
Upgrade from v2 linear row partition to Grassmannian subspace assignment.

v2: each domain owns contiguous rows [d*H_D : (d+1)*H_D].
    Capacity per domain: (H_D/H)^2 = 1/D^2 of full hidden capacity.

v3: each domain owns a k-dimensional subspace V_d of R^H,
    represented by an orthonormal basis P_d in R^(H x k).
    Gradient gate: G_tilde = P_d P_d^T @ G @ P_d P_d^T
    Orthogonality enforced: QR re-orthogonalization post-step.
    Init: SVD round-robin on a randomly initialized reference weight.
    Capacity per domain: k*H (Grassmannian DoF) vs (H/D)^2 for v2.
    At k = H/D: capacity ratio v3/v2 = D (full factor recovered).

Same benchmark as v2 for direct comparison:
- 5 permuted MNIST tasks
- H_TOTAL = 640, k = H_TOTAL // N_TASKS = 128 per domain
- 5 epochs per task
- Metric: accuracy on all prior tasks after each new task trained
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
N_TASKS     = 5
K           = H_TOTAL // N_TASKS   # subspace dim per domain = 128
N_CLASSES   = 10
INPUT_DIM   = 784
EPOCHS      = 5
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42
QR_EVERY    = 1   # re-orthogonalize every N steps (1 = every step, safe default)

# ── Permuted MNIST (same as v2) ───────────────────────────────────────────────

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


def get_loaders(base_train, base_test, permutation):
    train_ds = PermutedMNIST(base_train, permutation)
    test_ds  = PermutedMNIST(base_test,  permutation)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── SVD Round-Robin Initialization ───────────────────────────────────────────

def svd_roundrobin_init(n_domains, h, k, seed=SEED):
    """
    Initialize D mutually orthogonal basis matrices via SVD round-robin.

    1. Generate a random H x H reference matrix (stand-in for a pretrained W).
    2. Compute SVD: W = U S V^T. U is orthogonal, columns are left singular vectors.
    3. Distribute columns of U round-robin across domains:
       domain d gets columns {d, d+D, d+2D, ..., d+(k-1)*D}
    4. Each P_d = [u_{d}, u_{d+D}, ...] in R^(H x k).

    Guarantees: P_i^T P_j = 0 for i != j (disjoint subsets of orthonormal U).
    """
    torch.manual_seed(seed)
    W_ref = torch.randn(h, h)
    U, _, _ = torch.linalg.svd(W_ref, full_matrices=True)  # U: H x H, orthogonal

    bases = []
    for d in range(n_domains):
        indices = [d + j * n_domains for j in range(k)]
        P_d = U[:, indices]  # H x k, orthonormal columns
        bases.append(P_d)

    # Verify mutual orthogonality at init
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            overlap = (bases[i].T @ bases[j]).abs().max().item()
            assert overlap < 1e-5, f"Init orthogonality violated: domains {i},{j} overlap={overlap:.2e}"

    print(f"  SVD round-robin init: {n_domains} domains, H={h}, k={k}. Orthogonality verified.")
    return bases  # list of H x k tensors

# ── Grassmannian Layer ────────────────────────────────────────────────────────

class GrassmannianLayer(nn.Module):
    """
    A hidden layer where each domain owns a k-dimensional subspace of R^H.

    Forward pass for domain d:
        h = relu(W x)           -- full H x INPUT (or H x H) linear
        h_gated = Pi_d h        -- project onto V_d: Pi_d = P_d P_d^T

    Gradient gate (implemented via backward hook):
        G_tilde = Pi_d G Pi_d   -- zero out gradient components outside V_d

    P_d matrices are not nn.Parameters -- they are fixed orthonormal frames.
    They live as buffers, updated only by QR re-orthogonalization.
    """

    def __init__(self, in_features, out_features, bases):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_domains    = len(bases)
        self.k            = bases[0].shape[1]

        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Register basis matrices as buffers (not trained, not in optimizer)
        for d, P in enumerate(bases):
            self.register_buffer(f"P_{d}", P.clone().float())

        self._active_domain = None
        self._hook_handle   = None

    def get_P(self, d):
        return getattr(self, f"P_{d}")

    def set_active_domain(self, d):
        """Call before forward pass to set which domain is active."""
        self._active_domain = d
        # Register gradient hook on weight
        if self._hook_handle is not None:
            self._hook_handle.remove()
        P = self.get_P(d)
        Pi = P @ P.T  # H x H projection matrix

        def grad_hook(grad):
            # G_tilde = Pi @ G @ Pi  (project both input and output subspace)
            return Pi @ grad @ Pi

        self._hook_handle = self.linear.weight.register_hook(grad_hook)

    def forward(self, x):
        h = self.linear(x)                      # full linear: (batch, H)
        P = self.get_P(self._active_domain)
        Pi = P @ P.T                            # H x H
        h_gated = h @ Pi.T                     # project activations onto V_d
        return h_gated

    def reorthogonalize(self):
        """QR re-orthogonalization of all basis matrices."""
        P_all = torch.cat([self.get_P(d) for d in range(self.n_domains)], dim=1)  # H x (D*k)
        Q, _ = torch.linalg.qr(P_all)  # Q: H x (D*k), orthonormal columns
        for d in range(self.n_domains):
            P_d = Q[:, d * self.k : (d + 1) * self.k]
            getattr(self, f"P_{d}").copy_(P_d)

# ── PWP Grassmannian MLP ──────────────────────────────────────────────────────

class GrassmannianMLP(nn.Module):
    """
    Two Grassmannian hidden layers + per-domain output heads.
    All domains share the same weight tensors but operate in orthogonal subspaces.
    """

    def __init__(self, n_domains=N_TASKS, hidden=H_TOTAL, k=K, n_classes=N_CLASSES):
        super().__init__()
        self.n_domains = n_domains
        self.hidden    = hidden
        self.k         = k

        print(f"\nInitializing GrassmannianMLP: D={n_domains}, H={hidden}, k={k}")
        bases1 = svd_roundrobin_init(n_domains, hidden, k, seed=SEED)
        bases2 = svd_roundrobin_init(n_domains, hidden, k, seed=SEED + 1)

        self.layer1 = GrassmannianLayer(INPUT_DIM, hidden, bases1)
        self.layer2 = GrassmannianLayer(hidden,    hidden, bases2)
        self.relu   = nn.ReLU()

        # Private output head per domain
        self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_domains)])

    def forward(self, x, domain_id):
        self.layer1.set_active_domain(domain_id)
        self.layer2.set_active_domain(domain_id)
        h1 = self.relu(self.layer1(x))
        h2 = self.relu(self.layer2(h1))
        return self.heads[domain_id](h2)

    def reorthogonalize(self):
        self.layer1.reorthogonalize()
        self.layer2.reorthogonalize()

    def parameters_for_domain(self, domain_id):
        """Return only the active domain's head params + shared layer params."""
        return list(self.layer1.linear.parameters()) + \
               list(self.layer2.linear.parameters()) + \
               list(self.heads[domain_id].parameters())

# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, domain_id, step_counter):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, domain_id)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

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
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, domain_id)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── Main ──────────────────────────────────────────────────────────────────────

def run_grassmannian(perms, base_train, base_test):
    print("\n── PWP v3 Grassmannian ───────────────────────────────")
    model = GrassmannianMLP().to(DEVICE)

    test_loaders = []
    acc_matrix   = np.zeros((N_TASKS, N_TASKS))
    step_counter = [0]

    for task_id in range(N_TASKS):
        train_loader, test_loader = get_loaders(base_train, base_test, perms[task_id])
        test_loaders.append(test_loader)

        optimizer = optim.Adam(model.parameters_for_domain(task_id), lr=LR)

        print(f"\n  Training task {task_id} (k={K} subspace dims) ...")
        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, task_id, step_counter)
            print(f"    epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

        for prev_id in range(task_id + 1):
            acc = evaluate(model, test_loaders[prev_id], domain_id=prev_id)
            acc_matrix[task_id][prev_id] = acc
            print(f"  acc on task {prev_id} after training task {task_id}: {acc:.4f}")

    return acc_matrix


def print_comparison(v2_acc, v3_acc):
    print("\n\n══ v2 vs v3 COMPARISON ══════════════════════════════")
    print(f"{'Task':<6} {'v2 (end)':<16} {'v3 (end)':<16} {'Delta (v3-v2)':>14}")
    print("─" * 54)
    for task_id in range(N_TASKS):
        v2 = v2_acc[N_TASKS - 1][task_id]
        v3 = v3_acc[N_TASKS - 1][task_id]
        delta = v3 - v2
        print(f"  {task_id:<4} {v2:<16.4f} {v3:<16.4f} {delta:>+14.4f}")

    v2_mean = np.mean([v2_acc[N_TASKS-1][i] for i in range(N_TASKS)])
    v3_mean = np.mean([v3_acc[N_TASKS-1][i] for i in range(N_TASKS)])
    print(f"\n  v2 mean: {v2_mean:.4f}")
    print(f"  v3 mean: {v3_mean:.4f}")
    print(f"  Delta:   {v3_mean - v2_mean:+.4f}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    base_test  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    perms = make_permutations(N_TASKS)

    # Load v2 results for comparison if they exist
    try:
        v2_acc = np.load("baseline_acc_matrix.npy")
        # v2 results are in pwp_acc_matrix.npy from the v2 run
        v2_pwp_acc = np.load("pwp_acc_matrix.npy")
        has_v2 = True
        print("Loaded v2 results for comparison.")
    except FileNotFoundError:
        has_v2 = False
        print("No v2 results found. Running standalone.")

    v3_acc = run_grassmannian(perms, base_train, base_test)

    if has_v2:
        print_comparison(v2_pwp_acc, v3_acc)

    np.save("grassmannian_acc_matrix.npy", v3_acc)
    print("\nResults saved: grassmannian_acc_matrix.npy")


if __name__ == "__main__":
    main()