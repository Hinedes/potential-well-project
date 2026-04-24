"""
PWP v2 -- Linear Partition: Permuted MNIST
==========================================
Tests whether neuron-group block-diagonal partition holds
beyond trivially separable data (Split-MNIST).

Each task = MNIST with a fixed random pixel permutation applied.
Tasks are not class-disjoint -- all 10 digits appear in every task.
The only thing separating them is the permutation.
This is a harder test: the model cannot cheat by class routing.

Setup:
- 5 tasks, each a distinct permutation of MNIST pixels
- Baseline: standard MLP, sequential training (catastrophic forgetting expected)
- PWP: neuron-group block-diagonal MLP, D domains
- Metric: accuracy on task k after training on all tasks 1..N

Architecture (both):
- fc1: H_total neurons (input 784)
- fc2: H_total neurons
- Output: 10 classes (shared across tasks -- permuted MNIST uses same label space)

PWP partition:
- fc1 rows split into D equal blocks, one per domain
- fc2 block-diagonal enforced via forward-pass slicing
- Private output head per domain
- Gradient isolation by construction (autograd sees only active block)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from copy import deepcopy

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_TOTAL     = 640      # total hidden dim (shared between baseline and PWP)
N_TASKS     = 5        # number of permutation tasks
N_CLASSES   = 10
INPUT_DIM   = 784
EPOCHS      = 5        # per task
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42

# ── Permuted MNIST Dataset ────────────────────────────────────────────────────

class PermutedMNIST(Dataset):
    def __init__(self, base_dataset, permutation):
        self.base = base_dataset
        self.perm = permutation  # LongTensor of length 784

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = img.view(-1)[self.perm]  # apply permutation
        return img, label


def make_permutations(n_tasks, seed=SEED):
    rng = np.random.RandomState(seed)
    perms = [torch.arange(INPUT_DIM)]  # task 0 = identity (no permutation)
    for _ in range(n_tasks - 1):
        perms.append(torch.from_numpy(rng.permutation(INPUT_DIM)).long())
    return perms


def get_loaders(base_train, base_test, permutation):
    train_ds = PermutedMNIST(base_train, permutation)
    test_ds  = PermutedMNIST(base_test,  permutation)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── Baseline MLP ──────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    def __init__(self, hidden=H_TOTAL, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── PWP Block-Diagonal MLP ────────────────────────────────────────────────────

class PWPBlock(nn.Module):
    """
    Single domain block.
    fc1: INPUT_DIM -> H_D  (only rows belonging to this domain)
    fc2: H_D -> H_D        (isolated by forward-pass slicing; autograd gates gradient)
    head: H_D -> N_CLASSES
    """
    def __init__(self, h_d, n_classes=N_CLASSES):
        super().__init__()
        self.fc1  = nn.Linear(INPUT_DIM, h_d)
        self.fc2  = nn.Linear(h_d, h_d)
        self.head = nn.Linear(h_d, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.head(x)


class PWPMLP(nn.Module):
    """
    D domains, each a PWPBlock. H_D = H_TOTAL // D per domain.
    During training only the active domain's block receives gradients.
    Other blocks are not touched by the optimizer.
    """
    def __init__(self, n_domains, hidden=H_TOTAL, n_classes=N_CLASSES):
        super().__init__()
        assert hidden % n_domains == 0, "H_TOTAL must be divisible by n_domains"
        self.n_domains = n_domains
        self.h_d = hidden // n_domains
        self.blocks = nn.ModuleList([
            PWPBlock(self.h_d, n_classes) for _ in range(n_domains)
        ])

    def forward(self, x, domain_id):
        return self.blocks[domain_id](x)

    def parameters_for_domain(self, domain_id):
        return self.blocks[domain_id].parameters()

# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, domain_id=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, domain_id) if domain_id is not None else model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, domain_id=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, domain_id) if domain_id is not None else model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── Main ──────────────────────────────────────────────────────────────────────

def run_baseline(perms, base_train, base_test):
    print("\n── Baseline MLP ──────────────────────────────────────")
    model = BaselineMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    test_loaders = []
    acc_matrix   = np.zeros((N_TASKS, N_TASKS))  # acc_matrix[trained_up_to][task]

    for task_id in range(N_TASKS):
        train_loader, test_loader = get_loaders(base_train, base_test, perms[task_id])
        test_loaders.append(test_loader)

        print(f"\n  Training task {task_id} ...")
        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer)
            print(f"    epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

        for prev_id in range(task_id + 1):
            acc = evaluate(model, test_loaders[prev_id])
            acc_matrix[task_id][prev_id] = acc
            print(f"  acc on task {prev_id} after training task {task_id}: {acc:.4f}")

    return acc_matrix


def run_pwp(perms, base_train, base_test):
    print("\n── PWP v2 Linear Partition ───────────────────────────")
    # One domain per task; if N_TASKS > H_TOTAL we'd hit a floor -- not an issue here
    model = PWPMLP(n_domains=N_TASKS).to(DEVICE)

    test_loaders = []
    acc_matrix   = np.zeros((N_TASKS, N_TASKS))

    for task_id in range(N_TASKS):
        train_loader, test_loader = get_loaders(base_train, base_test, perms[task_id])
        test_loaders.append(test_loader)

        # Only optimize the active domain's parameters
        optimizer = optim.Adam(model.parameters_for_domain(task_id), lr=LR)

        print(f"\n  Training task {task_id} (domain {task_id}, H_D={model.h_d}) ...")
        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, domain_id=task_id)
            print(f"    epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

        for prev_id in range(task_id + 1):
            acc = evaluate(model, test_loaders[prev_id], domain_id=prev_id)
            acc_matrix[task_id][prev_id] = acc
            print(f"  acc on task {prev_id} after training task {task_id}: {acc:.4f}")

    return acc_matrix


def print_summary(baseline_acc, pwp_acc):
    print("\n\n══ RESULTS SUMMARY ══════════════════════════════════")
    print(f"{'Task':<6} {'Baseline (end)':<20} {'PWP (end)':<20} {'Delta':>8}")
    print("─" * 56)
    for task_id in range(N_TASKS):
        b = baseline_acc[N_TASKS - 1][task_id]
        p = pwp_acc[N_TASKS - 1][task_id]
        delta = b - p  # positive = baseline retained more (bad for PWP)
        print(f"  {task_id:<4} {b:<20.4f} {p:<20.4f} {delta:>+8.4f}")

    print("\nFinal accuracy after all tasks trained:")
    b_mean = np.mean([baseline_acc[N_TASKS-1][i] for i in range(N_TASKS)])
    p_mean = np.mean([pwp_acc[N_TASKS-1][i]      for i in range(N_TASKS)])
    print(f"  Baseline mean: {b_mean:.4f}")
    print(f"  PWP mean:      {p_mean:.4f}")
    print(f"  Delta:         {b_mean - p_mean:+.4f}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    base_test  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    perms = make_permutations(N_TASKS)

    baseline_acc = run_baseline(perms, base_train, base_test)
    pwp_acc      = run_pwp(perms, base_train, base_test)

    print_summary(baseline_acc, pwp_acc)

    # Save raw matrices for logging
    np.save("baseline_acc_matrix.npy", baseline_acc)
    np.save("pwp_acc_matrix.npy",      pwp_acc)
    print("\nAccuracy matrices saved: baseline_acc_matrix.npy, pwp_acc_matrix.npy")


if __name__ == "__main__":
    main()