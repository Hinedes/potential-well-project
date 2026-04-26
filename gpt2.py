"""
PWP on GPT-2 Small
==================

This is a working GPT-2 proof-of-concept for the Potential Well Project.

What it does:
- patches every GPT-2 MLP block with a PWP block,
- keeps domain 0 as the frozen base model,
- trains domain 1 inside an isolated subspace,
- checks whether domain 0 perplexity stays stable after domain 1 training.

This script follows the phase boundary from the project notes:
- k >= 64  -> Grassmannian
- k = 32   -> transition zone
- k <= 16  -> physical separation

The transition zone is left explicit. By default we choose the safer
physical mode there, but you can override it with FORCE_MODE.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformers.activations import ACT2FN
except ImportError as exc:
    raise SystemExit("Install transformers first: pip install transformers") from exc

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# -- Run config ----------------------------------------------------------------

MODEL_NAME = "gpt2"
N_DOMAINS = 2
TRAIN_DOMAIN = 1
SEQ_LEN = 512
BATCH_SIZE = 4
LR = 1e-4
TRAIN_STEPS = 200
EVAL_TOKENS = 8192
SEED = 42

EVAL_TEXT_FILE: Optional[str] = None
TRAIN_TEXT_FILE: Optional[str] = None

EVAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "test", 100_000)
TRAIN_DATASET = ("wikitext", "wikitext-103-raw-v1", "train", 200_000)

FORCE_MODE: Optional[str] = None
TRANSITION_DEFAULT = "physical"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context():
    if DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def select_architecture(
    hidden_size: int,
    n_domains: int,
    force_mode: Optional[str] = None,
    transition_default: str = TRANSITION_DEFAULT,
):
    if hidden_size % n_domains != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by n_domains={n_domains}"
        )

    k = hidden_size // n_domains

    if force_mode is not None:
        if force_mode not in {"grassmannian", "physical"}:
            raise ValueError("FORCE_MODE must be 'grassmannian' or 'physical'")
        return force_mode, k, f"forced via FORCE_MODE ({force_mode})"

    if k >= 64:
        return "grassmannian", k, "k >= 64"
    if k <= 16:
        return "physical", k, "k <= 16"
    return transition_default, k, f"transition zone (16 < k < 64), defaulting to {transition_default}"


def round_robin_bases(dim: int, n_domains: int, k: int, seed: int):
    if n_domains * k > dim:
        raise ValueError(f"Cannot fit {n_domains} domains with k={k} into dim={dim}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    q, _ = torch.linalg.qr(
        torch.randn(dim, n_domains * k, generator=generator, dtype=torch.float32),
        mode="reduced",
    )
    return [q[:, d : n_domains * k : n_domains].contiguous() for d in range(n_domains)]


def orthogonalize_against_frozen(basis: torch.Tensor, frozen_bases: list[torch.Tensor]):
    updated = basis.clone()
    for frozen in frozen_bases:
        updated = updated - frozen @ (frozen.T @ updated)

    q, r = torch.linalg.qr(updated, mode="reduced")
    diag = r.diag().abs()
    if diag.numel() and float(diag.min()) < 1e-6:
        raise RuntimeError(
            "Basis collapsed while orthogonalizing against frozen domains. "
            "Reduce N_DOMAINS or switch to physical separation."
        )
    return q[:, : basis.shape[1]]


class TokenDataset(Dataset):
    def __init__(self, text: str, tokenizer: GPT2Tokenizer, seq_len: int):
        token_ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        if token_ids.numel() == 0:
            raise ValueError("TokenDataset received empty text.")

        if token_ids.numel() < seq_len:
            pad = torch.full(
                (seq_len - token_ids.numel(),),
                tokenizer.eos_token_id,
                dtype=token_ids.dtype,
            )
            token_ids = torch.cat([token_ids, pad], dim=0)

        usable = (token_ids.numel() // seq_len) * seq_len
        self.chunks = token_ids[:usable].view(-1, seq_len)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


def load_text_source(
    *,
    text_file: Optional[str],
    dataset_name: str,
    dataset_config: str,
    split: str,
    char_limit: int,
):
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8")
    else:
        if load_dataset is None:
            raise RuntimeError(
                "datasets is not installed. Either install it with "
                "`pip install datasets` or set TRAIN_TEXT_FILE / EVAL_TEXT_FILE."
            )
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        if "text" not in dataset.column_names:
            raise RuntimeError(f"Dataset {dataset_name}/{dataset_config} has no 'text' column")
        text = "\n".join(dataset["text"])

    if char_limit:
        text = text[:char_limit]

    if not text.strip():
        raise ValueError("Loaded text source is empty after trimming.")

    return text


class PWPMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        n_domains: int,
        activation_name: str,
        layer_norm_eps: float,
        resid_pdrop: float,
        mode: str,
        seed: int,
    ):
        super().__init__()

        if hidden_size % n_domains != 0:
            raise ValueError("hidden_size must be divisible by n_domains")
        if intermediate_size % n_domains != 0:
            raise ValueError("intermediate_size must be divisible by n_domains")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_domains = n_domains
        self.mode = mode
        self.k_hidden = hidden_size // n_domains
        self.k_mid = intermediate_size // n_domains

        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = ACT2FN[activation_name]
        self.dropout = nn.Dropout(resid_pdrop)

        self.frozen: set[int] = set()
        self._active = 0
        self._hooks = []

        self._register_domain_state("in", hidden_size, self.k_hidden, seed + 11)
        self._register_domain_state("mid", intermediate_size, self.k_mid, seed + 23)
        self._register_domain_state("out", hidden_size, self.k_hidden, seed + 37)

    def _register_domain_state(self, tag: str, dim: int, k: int, seed: int):
        if self.mode == "grassmannian":
            bases = round_robin_bases(dim, self.n_domains, k, seed)
            for domain_id, basis in enumerate(bases):
                self.register_buffer(f"{tag}_state_{domain_id}", basis)
            return

        for domain_id in range(self.n_domains):
            mask = torch.zeros(dim, dtype=torch.float32)
            start = domain_id * k
            stop = start + k
            mask[start:stop] = 1.0
            self.register_buffer(f"{tag}_state_{domain_id}", mask)

    def _state(self, tag: str, domain_id: int):
        return getattr(self, f"{tag}_state_{domain_id}")

    def _project_features(self, tensor: torch.Tensor, tag: str, domain_id: int):
        state = self._state(tag, domain_id)
        if self.mode == "grassmannian":
            basis = state.to(device=tensor.device, dtype=tensor.dtype)
            return (tensor @ basis) @ basis.T
        mask = state.to(device=tensor.device, dtype=tensor.dtype)
        return tensor * mask

    def _project_vector(self, grad: torch.Tensor, tag: str, domain_id: int):
        state = self._state(tag, domain_id)
        if self.mode == "grassmannian":
            basis = state.to(device=grad.device, dtype=grad.dtype)
            return basis @ (basis.T @ grad)
        mask = state.to(device=grad.device, dtype=grad.dtype)
        return grad * mask

    def _project_matrix(
        self,
        grad: torch.Tensor,
        row_tag: str,
        col_tag: str,
        domain_id: int,
    ):
        row_state = self._state(row_tag, domain_id)
        col_state = self._state(col_tag, domain_id)

        if self.mode == "grassmannian":
            row_basis = row_state.to(device=grad.device, dtype=grad.dtype)
            col_basis = col_state.to(device=grad.device, dtype=grad.dtype)
            return row_basis @ (row_basis.T @ grad @ col_basis) @ col_basis.T

        row_mask = row_state.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
        col_mask = col_state.to(device=grad.device, dtype=grad.dtype).unsqueeze(0)
        return grad * row_mask * col_mask

    def domain_parameters(self):
        yield from self.ln.parameters()
        yield from self.fc1.parameters()
        yield from self.fc2.parameters()

    def prepare_domain(self, domain_id: int):
        if self.mode != "grassmannian":
            return

        with torch.no_grad():
            for tag in ("in", "mid", "out"):
                frozen_bases = [
                    self._state(tag, frozen_id).detach()
                    for frozen_id in sorted(self.frozen)
                ]
                if not frozen_bases:
                    continue
                updated = orthogonalize_against_frozen(
                    self._state(tag, domain_id), frozen_bases
                )
                self._state(tag, domain_id).copy_(updated)

    def freeze_domain(self, domain_id: int):
        self.frozen.add(domain_id)

    def set_active(self, domain_id: int):
        self._active = domain_id
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        self._hooks.append(
            self.ln.weight.register_hook(
                lambda grad, d=domain_id: self._project_vector(grad, "in", d)
            )
        )
        self._hooks.append(
            self.ln.bias.register_hook(
                lambda grad, d=domain_id: self._project_vector(grad, "in", d)
            )
        )
        self._hooks.append(
            self.fc1.weight.register_hook(
                lambda grad, d=domain_id: self._project_matrix(grad, "mid", "in", d)
            )
        )
        self._hooks.append(
            self.fc1.bias.register_hook(
                lambda grad, d=domain_id: self._project_vector(grad, "mid", d)
            )
        )
        self._hooks.append(
            self.fc2.weight.register_hook(
                lambda grad, d=domain_id: self._project_matrix(grad, "out", "mid", d)
            )
        )
        self._hooks.append(
            self.fc2.bias.register_hook(
                lambda grad, d=domain_id: self._project_vector(grad, "out", d)
            )
        )

    def forward(self, x: torch.Tensor):
        x = self.ln(x)
        x = self._project_features(x, "in", self._active)
        x = self.fc1(x)
        x = self._project_features(x, "mid", self._active)
        x = self.act(x)
        x = self._project_features(x, "mid", self._active)
        x = self.fc2(x)
        x = self._project_features(x, "out", self._active)
        return self.dropout(x)


def iter_pwp_blocks(model: GPT2LMHeadModel):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLPBlock):
            yield block.mlp


def patch_gpt2(model: GPT2LMHeadModel):
    hidden_size = model.config.n_embd
    intermediate_size = model.config.n_inner or (4 * hidden_size)
    mode, k, reason = select_architecture(hidden_size, N_DOMAINS, FORCE_MODE)

    print(
        f"\nPatching GPT-2 ({len(model.transformer.h)} layers) "
        f"with mode={mode}, k={k} [{reason}]..."
    )

    for layer_idx, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, PWPMLPBlock):
            continue

        source_mlp = block.mlp
        source_ln = block.ln_2

        pwp = PWPMLPBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_domains=N_DOMAINS,
            activation_name=model.config.activation_function,
            layer_norm_eps=model.config.layer_norm_epsilon,
            resid_pdrop=model.config.resid_pdrop,
            mode=mode,
            seed=SEED + layer_idx * 101,
        ).to(device=source_ln.weight.device)

        with torch.no_grad():
            pwp.ln.weight.copy_(source_ln.weight.float())
            pwp.ln.bias.copy_(source_ln.bias.float())
            pwp.fc1.weight.copy_(source_mlp.c_fc.weight.t().float())
            pwp.fc1.bias.copy_(source_mlp.c_fc.bias.float())
            pwp.fc2.weight.copy_(source_mlp.c_proj.weight.t().float())
            pwp.fc2.bias.copy_(source_mlp.c_proj.bias.float())

        block.mlp = pwp
        block.ln_2 = nn.Identity()

    return model, mode, k, reason


def set_active_domain(model: GPT2LMHeadModel, domain_id: int):
    for pwp in iter_pwp_blocks(model):
        pwp.set_active(domain_id)


def prepare_domain(model: GPT2LMHeadModel, domain_id: int):
    for pwp in iter_pwp_blocks(model):
        pwp.prepare_domain(domain_id)


def freeze_domain(model: GPT2LMHeadModel, domain_id: int):
    for pwp in iter_pwp_blocks(model):
        pwp.freeze_domain(domain_id)


def configure_pwp_training(model: GPT2LMHeadModel):
    for param in model.parameters():
        param.requires_grad = False

    trainable = []
    for pwp in iter_pwp_blocks(model):
        for param in pwp.domain_parameters():
            param.requires_grad = True
            trainable.append(param)
    return trainable


@torch.no_grad()
def compute_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    *,
    domain_id: Optional[int],
):
    model.eval()
    if domain_id is not None:
        set_active_domain(model, domain_id)

    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    tokens = tokens[:EVAL_TOKENS].to(DEVICE)
    if tokens.numel() < 2:
        raise ValueError("Need at least two tokens to compute perplexity.")

    nlls = []
    stride = max(SEQ_LEN // 2, 1)

    if tokens.numel() <= SEQ_LEN:
        windows = [tokens]
    else:
        windows = [
            tokens[begin : min(begin + SEQ_LEN, tokens.numel())]
            for begin in range(0, tokens.numel() - 1, stride)
        ]

    for window in windows:
        chunk = window.unsqueeze(0)
        with amp_context():
            loss = model(chunk, labels=chunk.clone()).loss.float()
        nlls.append(loss)

    return torch.exp(torch.stack(nlls).mean()).item()


def train_domain(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    *,
    domain_id: int,
):
    print(f"\nTraining domain {domain_id} ({TRAIN_STEPS} steps)...")

    prepare_domain(model, domain_id)
    set_active_domain(model, domain_id)
    model.train()
    model.config.use_cache = False

    trainable_params = configure_pwp_training(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    loader = DataLoader(
        TokenDataset(text, tokenizer, SEQ_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    step = 0
    while step < TRAIN_STEPS:
        for batch in loader:
            if step >= TRAIN_STEPS:
                break

            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                loss = model(batch, labels=batch.clone()).loss
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  step {step:>4}/{TRAIN_STEPS}  loss={loss.item():.4f}")
            step += 1

    freeze_domain(model, domain_id)
    print(f"  Domain {domain_id} training complete.")


def main():
    if TRAIN_DOMAIN >= N_DOMAINS:
        raise ValueError("TRAIN_DOMAIN must be smaller than N_DOMAINS")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Loading text sources...")
    eval_text = load_text_source(
        text_file=EVAL_TEXT_FILE,
        dataset_name=EVAL_DATASET[0],
        dataset_config=EVAL_DATASET[1],
        split=EVAL_DATASET[2],
        char_limit=EVAL_DATASET[3],
    )
    train_text = load_text_source(
        text_file=TRAIN_TEXT_FILE,
        dataset_name=TRAIN_DATASET[0],
        dataset_config=TRAIN_DATASET[1],
        split=TRAIN_DATASET[2],
        char_limit=TRAIN_DATASET[3],
    )

    print("\nBaseline PPL (unpatched)...")
    baseline_ppl = compute_perplexity(model, tokenizer, eval_text, domain_id=None)
    print(f"  Eval PPL: {baseline_ppl:.3f}")

    model, mode, k, reason = patch_gpt2(model)
    print(f"  Selected mode: {mode} (k={k}, {reason})")

    prepare_domain(model, 0)
    set_active_domain(model, 0)
    freeze_domain(model, 0)

    print("\nPost-patch PPL (domain 0)...")
    post_patch_ppl = compute_perplexity(model, tokenizer, eval_text, domain_id=0)
    print(f"  Eval PPL: {post_patch_ppl:.3f}")
    print(f"  Delta from baseline: {post_patch_ppl - baseline_ppl:+.3f}")

    train_domain(model, tokenizer, train_text, domain_id=TRAIN_DOMAIN)

    print("\nFinal PPL (domain 0, after domain 1)...")
    final_ppl = compute_perplexity(model, tokenizer, eval_text, domain_id=0)
    print(f"  Eval PPL: {final_ppl:.3f}")

    delta = final_ppl - post_patch_ppl
    status = "PASS" if abs(delta) < 0.5 else "FAIL"

    print("\n== RESULT ===============================")
    print(f"  Baseline PPL:       {baseline_ppl:.3f}")
    print(f"  Post-patch PPL:     {post_patch_ppl:.3f}")
    print(f"  After domain 1 PPL: {final_ppl:.3f}")
    print(f"  Retention delta:    {delta:+.3f}  [{status}]")

    np.save(
        "gpt2_pwp_results.npy",
        np.array([baseline_ppl, post_patch_ppl, final_ppl], dtype=np.float32),
    )
    print("Saved: gpt2_pwp_results.npy")


if __name__ == "__main__":
    main()
