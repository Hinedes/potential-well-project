"""
PWP on GPT-2 Small -- First Real Model Test
============================================
Replaces GPT-2's MLP linear layers with PWP-partitioned te.Linear layers.
Uses Transformer Engine (te) instead of nn for linear layers.

Test:
    Domain 0 = original GPT-2 weights, measured by perplexity on WikiText-2
    Domain 1 = new fine-tuning domain (WikiText-103 subset, different distribution)
    Metric:   perplexity on WikiText-2 test set before and after domain 1 training
    Pass:     perplexity delta < 0.5 (domain 0 sector untouched by construction)

Architecture:
    GPT-2 small: hidden_size=768, n_layer=12, ffn_dim=3072
    PWP patches the MLP sublayer in each transformer block:
        c_fc:   768  -> 3072  (fan-out)
        c_proj: 3072 -> 768   (fan-in)
    Attention layers left unmodified for this experiment.

Mode selector (empirically derived from sweep):
    k = hidden_size // D
    k >= 64  ->  Grassmannian (subspace projection)
    k <  64  ->  Physical separation (block diagonal)

At D=2: k=384 -> Grassmannian.
At D=4: k=192 -> Grassmannian.
At D=8: k=96  -> Grassmannian.

Transformer Engine notes:
    te.Linear provides fused GEMM kernels optimized for Blackwell (sm_120).
    FP8 autocast is optional -- disabled by default here for stability.
    Gradient hooks attach to te.Linear.weight, same API as nn.Linear.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
import copy

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
    print("Transformer Engine found. Using te.Linear.")
except ImportError:
    HAS_TE = False
    print("Transformer Engine not found. Falling back to nn.Linear.")
    te = nn  # shim: te.Linear -> nn.Linear

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("ERROR: transformers and datasets required.")
    print("  pip install transformers datasets")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DOMAINS   = 2          # domain 0 = original GPT-2, domain 1 = new domain
HIDDEN      = 768        # GPT-2 small hidden_size
FFN_DIM     = 3072       # GPT-2 small ffn intermediate dim
SEQ_LEN     = 512        # context window for perplexity eval
BATCH_SIZE  = 4
LR          = 1e-4
TRAIN_STEPS = 200        # domain 1 fine-tuning steps (short -- just enough to move weights)
EVAL_TOKENS = 4096       # tokens to evaluate perplexity over
SEED        = 42
USE_FP8     = False      # set True if H100/H200/Blackwell + TE FP8 confirmed working

K = HIDDEN // N_DOMAINS  # 384 -- well above 64, Grassmannian mode

# ── Mode selector ─────────────────────────────────────────────────────────────

def select_mode(k):
    return "grassmannian" if k >= 64 else "physical"

print(f"\nMode: {select_mode(K)} (k={K}, D={N_DOMAINS}, H={HIDDEN})")

# ── SVD round-robin init ──────────────────────────────────────────────────────

def svd_roundrobin(n, h, k, seed=SEED):
    torch.manual_seed(seed)
    U, _, _ = torch.linalg.svd(torch.randn(h, h), full_matrices=True)
    bases = []
    for d in range(n):
        idx = [d + j * n for j in range(k)]
        bases.append(U[:, idx].clone().float())
    return bases

# ── PWP Linear Layer (te.Linear wrapper) ─────────────────────────────────────

class PWPLinear(nn.Module):
    """
    Drop-in replacement for a linear layer with PWP domain partitioning.

    Uses te.Linear as the underlying compute primitive when TE is available.
    Gradient hook gates updates to the active domain's subspace.
    Domain-local incremental Gram-Schmidt enforces orthogonality at domain init.

    Mode: Grassmannian (k >= 64) or Physical (k < 64) selected automatically.
    """

    def __init__(self, in_features, out_features, n_domains, bias=True,
                 seed_offset=0, layer_id=0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_domains    = n_domains
        self.k            = out_features // n_domains
        self.mode         = select_mode(self.k)
        self.frozen       = set()
        self._active      = None
        self._hook        = None

        # Core linear layer -- te.Linear if available, else nn.Linear
        if HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        if self.mode == "grassmannian":
            # Grassmannian: orthonormal basis per domain
            bases = svd_roundrobin(n_domains, out_features, self.k,
                                   seed=SEED + seed_offset + layer_id * 7)
            for d, P in enumerate(bases):
                self.register_buffer(f"P_{d}", P)
        else:
            # Physical: row partition, no basis matrices needed
            self.h_d = self.k  # out_features // n_domains rows per domain

    def get_P(self, d):
        return getattr(self, f"P_{d}")

    def prepare_domain(self, d):
        """
        Incremental Gram-Schmidt: project P_d into null space of frozen domains.
        Only meaningful in Grassmannian mode.
        """
        if self.mode != "grassmannian":
            return
        frozen_bases = [self.get_P(f).detach() for f in sorted(self.frozen)]
        if not frozen_bases:
            return
        V = self.get_P(d).clone()
        for P_f in frozen_bases:
            V = V - P_f @ (P_f.T @ V)
        Q, R = torch.linalg.qr(V)
        diag = R.diag().abs()
        if (diag < 1e-6).any():
            print(f"    WARNING layer {d}: subspace near-collapse, "
                  f"min diag={diag.min():.2e}")
        self.get_P(d).copy_(Q)

    def freeze_domain(self, d):
        self.frozen.add(d)

    def set_active(self, d):
        self._active = d
        if self._hook:
            self._hook.remove()

        if self.mode == "grassmannian":
            P  = self.get_P(d)
            Pi = P @ P.T

            def hook(g):
                # Project gradient onto domain d's subspace
                projected = Pi @ g
                if g.shape[1] == Pi.shape[0]:
                    projected = projected @ Pi
                return projected

        else:
            # Physical: zero out gradients outside domain d's row block
            start = d * self.h_d
            end   = start + self.h_d

            def hook(g):
                mask = torch.zeros_like(g)
                mask[start:end] = g[start:end]
                return mask

        self._hook = self.linear.weight.register_hook(hook)

    def forward(self, x):
        h = self.linear(x)   # (batch, seq, out_features) or (batch, out_features)

        if self.mode == "grassmannian" and self._active is not None:
            P  = self.get_P(self._active)
            Pi = P @ P.T
            h  = h @ Pi.T    # project activations onto active domain's subspace

        elif self.mode == "physical" and self._active is not None:
            # Zero out activations outside domain d's rows
            mask = torch.zeros_like(h)
            start = self._active * self.h_d
            end   = start + self.h_d
            mask[..., start:end] = h[..., start:end]
            h = mask

        return h


# ── PWP MLP block (replaces GPT-2's MLP sublayer) ────────────────────────────

class PWPMLP(nn.Module):
    """
    Replaces GPT-2's MLP sublayer.
    GPT-2 MLP: c_fc (768->3072) + gelu + c_proj (3072->768)
    """

    def __init__(self, n_domains=N_DOMAINS, hidden=HIDDEN, ffn_dim=FFN_DIM):
        super().__init__()
        self.c_fc   = PWPLinear(hidden, ffn_dim, n_domains, seed_offset=0)
        self.c_proj = PWPLinear(ffn_dim, hidden, n_domains, seed_offset=50)
        self.act    = nn.GELU()

    def set_active(self, d):
        self.c_fc.set_active(d)
        self.c_proj.set_active(d)

    def prepare_domain(self, d):
        self.c_fc.prepare_domain(d)
        self.c_proj.prepare_domain(d)

    def freeze_domain(self, d):
        self.c_fc.freeze_domain(d)
        self.c_proj.freeze_domain(d)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


# ── Patch GPT-2 ───────────────────────────────────────────────────────────────

def patch_gpt2(model, n_domains=N_DOMAINS):
    """
    Replace each transformer block's MLP sublayer with PWPMLP.
    Copy original weights into domain 0's partition.
    """
    print(f"\nPatching GPT-2 ({len(model.transformer.h)} layers) with PWP MLP...")

    for layer_idx, block in enumerate(model.transformer.h):
        orig_mlp = block.mlp
        pwp_mlp  = PWPMLP(n_domains=n_domains).to(DEVICE)

        # Copy original c_fc weights into the PWP linear's underlying weight
        with torch.no_grad():
            pwp_mlp.c_fc.linear.weight.copy_(orig_mlp.c_fc.weight)
            if orig_mlp.c_fc.bias is not None:
                pwp_mlp.c_fc.linear.bias.copy_(orig_mlp.c_fc.bias)

            pwp_mlp.c_proj.linear.weight.copy_(orig_mlp.c_proj.weight)
            if orig_mlp.c_proj.bias is not None:
                pwp_mlp.c_proj.linear.bias.copy_(orig_mlp.c_proj.bias)

        block.mlp = pwp_mlp

    print(f"  Patched. Mode: {select_mode(K)} (k={K})")
    return model


def set_active_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLP):
            block.mlp.set_active(d)


def prepare_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLP):
            block.mlp.prepare_domain(d)


def freeze_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLP):
            block.mlp.freeze_domain(d)


# ── Perplexity Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, domain_id=0, max_tokens=EVAL_TOKENS):
    """
    Compute perplexity over a text corpus using sliding window.
    Sets domain to domain_id before evaluation.
    """
    model.eval()
    set_active_domain(model, domain_id)

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0][:max_tokens].to(DEVICE)

    nlls   = []
    stride = SEQ_LEN // 2

    for begin in range(0, input_ids.size(0) - SEQ_LEN, stride):
        end      = begin + SEQ_LEN
        chunk    = input_ids[begin:end].unsqueeze(0)
        target   = chunk.clone()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out  = model(chunk, labels=target)
        nlls.append(out.loss.float())

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_text_corpus(dataset_name, split="train", max_chars=500_000):
    """Load a text corpus from HuggingFace datasets."""
    if dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(ds["text"])
    elif dataset_name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n".join(ds["text"])
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", split=split)
        text = "\n".join(ds["sentence"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return text[:max_chars]


class TokenDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=SEQ_LEN):
        tokens = tokenizer(text, return_tensors="pt",
                           truncation=False)["input_ids"][0]
        # Chunk into non-overlapping windows
        n = (len(tokens) // seq_len) * seq_len
        self.chunks = tokens[:n].view(-1, seq_len)

    def __len__(self):   return len(self.chunks)
    def __getitem__(self, i): return self.chunks[i]


# ── Training domain 1 ─────────────────────────────────────────────────────────

def train_domain1(model, tokenizer, corpus_text, steps=TRAIN_STEPS):
    """Fine-tune domain 1 on a new corpus. Domain 0 is frozen by construction."""
    print(f"\nTraining domain 1 ({steps} steps)...")
    prepare_domain(model, 1)
    set_active_domain(model, 1)

    ds     = TokenDataset(corpus_text, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Only optimize domain 1 MLP parameters (the shared linear weights)
    # In practice: optimizer sees all params but gradient hook zeros cross-domain updates
    params = []
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLP):
            params += list(block.mlp.c_fc.linear.parameters())
            params += list(block.mlp.c_proj.linear.parameters())

    opt     = torch.optim.AdamW(params, lr=LR)
    scaler  = torch.amp.GradScaler()
    step    = 0
    losses  = []

    model.train()
    for batch in loader:
        if step >= steps:
            break
        batch = batch.to(DEVICE)
        labels = batch.clone()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out  = model(batch, labels=labels)
            loss = out.loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}")
        step += 1

    freeze_domain(model, 1)
    print(f"  Domain 1 training complete. Mean loss: {np.mean(losses):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load GPT-2 small
    print("\nLoading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load corpora
    print("\nLoading corpora...")
    wt2_test  = get_text_corpus("wikitext-2",  split="test")
    wt103_train = get_text_corpus("wikitext-103", split="train", max_chars=200_000)
    print(f"  WikiText-2 test:    {len(wt2_test):,} chars")
    print(f"  WikiText-103 train: {len(wt103_train):,} chars (domain 1)")

    # Baseline perplexity (unpatched GPT-2)
    print("\nBaseline perplexity (unpatched GPT-2)...")
    # Temporarily measure before patching
    baseline_ppl = compute_perplexity(model, tokenizer, wt2_test,
                                      domain_id=None if not hasattr(model.transformer.h[0].mlp, 'set_active') else 0)
    print(f"  WikiText-2 PPL (baseline): {baseline_ppl:.3f}")

    # Patch GPT-2 with PWP MLP
    model = patch_gpt2(model, n_domains=N_DOMAINS)

    # Initialize domain 0 as active, prepare and freeze it
    prepare_domain(model, 0)   # no frozen bases yet -- domain 0 gets full space
    set_active_domain(model, 0)
    freeze_domain(model, 0)    # lock domain 0 permanently

    # Perplexity after patching, domain 0
    print("\nPerplexity after patching (domain 0, before domain 1 training)...")
    post_patch_ppl = compute_perplexity(model, tokenizer, wt2_test, domain_id=0)
    print(f"  WikiText-2 PPL (domain 0, patched): {post_patch_ppl:.3f}")
    print(f"  Delta from baseline: {post_patch_ppl - baseline_ppl:+.3f}")

    # Train domain 1
    train_domain1(model, tokenizer, wt103_train)

    # Perplexity after domain 1 training -- domain 0 should be unchanged
    print("\nPerplexity after domain 1 training (domain 0)...")
    final_ppl = compute_perplexity(model, tokenizer, wt2_test, domain_id=0)
    print(f"  WikiText-2 PPL (domain 0, after domain 1): {final_ppl:.3f}")
    print(f"  Delta from post-patch: {final_ppl - post_patch_ppl:+.3f}")
    print(f"  Delta from baseline:   {final_ppl - baseline_ppl:+.3f}")

    print("\n══ RESULT ══════════════════════════════════════════")
    print(f"  Baseline PPL:          {baseline_ppl:.3f}")
    print(f"  Post-patch PPL:        {post_patch_ppl:.3f}")
    print(f"  After domain 1 PPL:    {final_ppl:.3f}")
    retention_delta = final_ppl - post_patch_ppl
    status = "PASS" if abs(retention_delta) < 0.5 else "FAIL"
    print(f"  Retention delta:       {retention_delta:+.3f}  [{status}]")
    print(f"\n  {'Domain 0 retained.' if status == 'PASS' else 'Domain 0 degraded -- investigate.'}")

    np.save("gpt2_pwp_results.npy", np.array([baseline_ppl, post_patch_ppl, final_ppl]))
    print("\nSaved: gpt2_pwp_results.npy")


if __name__ == "__main__":
    main()