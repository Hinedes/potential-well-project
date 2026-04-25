"""
PWP on GPT-2 Small -- The Hypervisor Architecture (Transformer Engine)
====================================================================
This script implements the Potential Well Project (PWP) using a Ring -1 
Hypervisor model. 

- Ring 0: The physical GPT-2 weight tensors.
- Ring 3: The Guest OS (AdamW Optimizer), which believes it is training the whole network.
- Ring -1: The Grassmannian Projection Hooks, which intercept the optimizer's 
           memory writes and sandbox them into mutually orthogonal subspaces.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
    print("Transformer Engine available. Using te.LayerNormMLP + fp8_autocast.")
except ImportError:
    HAS_TE = False
    print("Transformer Engine not found. Falling back to nn.Linear.")

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from datasets import load_dataset
except ImportError:
    print("pip install transformers datasets")
    raise

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DOMAINS   = 2
HIDDEN      = 768
FFN_DIM     = 3072
SEQ_LEN     = 512
BATCH_SIZE  = 4
LR          = 1e-4
TRAIN_STEPS = 200
EVAL_TOKENS = 8192
SEED        = 42
USE_FP8     = True

K = HIDDEN // N_DOMAINS  # 384 -> Grassmannian mode

fp8_recipe = DelayedScaling(
    margin=0,
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
) if HAS_TE else None

# ── SVD round-robin init (The Memory Allocator) ───────────────────────────────

def svd_roundrobin(n, h, k, seed=SEED):
    torch.manual_seed(seed)
    U, _, _ = torch.linalg.svd(torch.randn(h, h), full_matrices=True)
    bases = []
    for d in range(n):
        idx = [d + j * n for j in range(k)]
        bases.append(U[:, idx].clone().float())
    return bases

# ── PWP MLP block (The Virtual Machine) ───────────────────────────────────────

class PWPMLPBlock(nn.Module):
    INCLUDES_LAYERNORM = True

    def __init__(self, hidden=HIDDEN, ffn_dim=FFN_DIM, n_domains=N_DOMAINS):
        super().__init__()
        self.n_domains = n_domains
        self.k         = hidden // n_domains

        if HAS_TE:
            self.mlp = te.LayerNormMLP(
                hidden_size=hidden,
                ffn_hidden_size=ffn_dim,
                eps=1e-5,
                bias=True,
                normalization="LayerNorm",
                activation="gelu",
                params_dtype=torch.bfloat16,
            )
        else:
            self.ln   = nn.LayerNorm(hidden)
            self.fc1  = nn.Linear(hidden, ffn_dim)
            self.fc2  = nn.Linear(ffn_dim, hidden)
            self.gelu = nn.GELU()

        bases_in  = svd_roundrobin(n_domains, hidden, self.k, seed=SEED)
        bases_out = svd_roundrobin(n_domains, hidden, self.k, seed=SEED + 1)
        for d in range(n_domains):
            self.register_buffer(f"P_in_{d}",  bases_in[d])
            self.register_buffer(f"P_out_{d}", bases_out[d])

        self.frozen      = set()
        self._active     = 0
        self._hooks      = []
        self._first_step = True

    def get_P_in(self, d):  return getattr(self, f"P_in_{d}")
    def get_P_out(self, d): return getattr(self, f"P_out_{d}")

    def set_active(self, d):
        self._active     = d
        self._first_step = True
        for h in self._hooks: h.remove()
        self._hooks = []
        self._install_hooks(d)

    def _install_hooks(self, d):
        """
        RING -1: The Hardware Trap
        This is the Controller. It intercepts the raw memory writes from the 
        optimizer and translates them using the MMU (Pi_in, Pi_out).
        """
        Pi_in  = self.get_P_in(d)  @ self.get_P_in(d).T
        Pi_out = self.get_P_out(d) @ self.get_P_out(d).T

        if HAS_TE:
            for name, param in self.mlp.named_parameters():
                if "fc1_weight" in name:
                    def fc1_hook(g, Pi=Pi_in): return g @ Pi
                    self._hooks.append(param.register_hook(fc1_hook))
                elif "fc2_weight" in name:
                    def fc2_hook(g, Pi=Pi_out): return Pi @ g
                    self._hooks.append(param.register_hook(fc2_hook))
        else:
            self._hooks.append(
                self.fc1.weight.register_hook(lambda g, Pi=Pi_in: g @ Pi))
            self._hooks.append(
                self.fc2.weight.register_hook(lambda g, Pi=Pi_out: Pi @ g))

    def prepare_domain(self, d):
        for attr in [f"P_in_{d}", f"P_out_{d}"]:
            tag = "in" if "_in_" in attr else "out"
            frozen_bases = [
                getattr(self, f"P_{tag}_{f}").detach()
                for f in sorted(self.frozen)
            ]
            if not frozen_bases:
                continue
            V = getattr(self, attr).clone()
            for P_f in frozen_bases:
                V = V - P_f @ (P_f.T @ V)
            Q, _ = torch.linalg.qr(V)
            getattr(self, attr).copy_(Q)

    def freeze_domain(self, d):
        self.frozen.add(d)

    def forward(self, x):
        Pi_in = self.get_P_in(self._active) @ self.get_P_in(self._active).T
        Pi_out = self.get_P_out(self._active) @ self.get_P_out(self._active).T

        # Isolate the input to the active domain's subspace
        x_proj = x @ Pi_in.T

        if HAS_TE:
            with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
                out = self.mlp(x_proj, is_first_microbatch=self._first_step)
            self._first_step = False
        else:
            out = self.fc2(self.gelu(self.fc1(self.ln(x_proj))))

        return out @ Pi_out.T   # project onto active domain's output subspace


# ── Patch GPT-2 ───────────────────────────────────────────────────────────────

def patch_gpt2(model):
    print(f"\nPatching GPT-2 ({len(model.transformer.h)} layers)...")
    for block in model.transformer.h:
        pwp = PWPMLPBlock().to(DEVICE)
        with torch.no_grad():
            if HAS_TE:
                sd = {n: p for n, p in pwp.mlp.named_parameters()}
                if "layer_norm_weight" in sd:
                    sd["layer_norm_weight"].copy_(block.ln_2.weight.to(torch.bfloat16))
                    sd["layer_norm_bias"].copy_(block.ln_2.bias.to(torch.bfloat16))
                if "fc1_weight" in sd:
                    sd["fc1_weight"].copy_(block.mlp.c_fc.weight.t().to(torch.bfloat16))
                if "fc1_bias" in sd:
                    sd["fc1_bias"].copy_(block.mlp.c_fc.bias.to(torch.bfloat16))
                if "fc2_weight" in sd:
                    sd["fc2_weight"].copy_(block.mlp.c_proj.weight.t().to(torch.bfloat16))
                if "fc2_bias" in sd:
                    sd["fc2_bias"].copy_(block.mlp.c_proj.bias.to(torch.bfloat16))
            else:
                pwp.ln.weight.copy_(block.ln_2.weight)
                pwp.ln.bias.copy_(block.ln_2.bias)
                pwp.fc1.weight.copy_(block.mlp.c_fc.weight.t())
                pwp.fc1.bias.copy_(block.mlp.c_fc.bias)
                pwp.fc2.weight.copy_(block.mlp.c_proj.weight.t())

    print(f"  Patched. k={K}, D={N_DOMAINS}, mode=Grassmannian")
    return model

def set_active_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLPBlock):
            block.mlp.set_active(d)

def prepare_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLPBlock):
            block.mlp.prepare_domain(d)

def freeze_domain(model, d):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLPBlock):
            block.mlp.freeze_domain(d)


# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, domain_id=0):
    model.eval()
    if domain_id is not None and any(
            isinstance(b.mlp, PWPMLPBlock) for b in model.transformer.h):
        set_active_domain(model, domain_id)

    tokens = tokenizer(text, return_tensors="pt",
                       truncation=False).input_ids[0][:EVAL_TOKENS].to(DEVICE)
    nlls, stride = [], SEQ_LEN // 2

    for begin in range(0, tokens.size(0) - SEQ_LEN, stride):
        chunk = tokens[begin:begin + SEQ_LEN].unsqueeze(0)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            nlls.append(model(chunk, labels=chunk.clone()).loss.float())

    return torch.exp(torch.stack(nlls).mean()).item()


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, text, tokenizer):
        ids = tokenizer(text, return_tensors="pt",
                        truncation=False).input_ids[0]
        n = (len(ids) // SEQ_LEN) * SEQ_LEN
        self.chunks = ids[:n].view(-1, SEQ_LEN)
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i): return self.chunks[i]


# ── Train domain 1 ────────────────────────────────────────────────────────────

def train_domain1(model, tokenizer, text):
    print(f"\nTraining domain 1 ({TRAIN_STEPS} steps)...")
    prepare_domain(model, 1)
    set_active_domain(model, 1)

    # ---------------------------------------------------------
    # RING -1: PROVISIONING GUEST OS MEMORY ACCESS
    # ---------------------------------------------------------
    params_for_optimizer = []
    
    for name, p in model.named_parameters():
        # Grant Ring 0 access to the core 2D matrices where the 
        # Grassmannian hypervisor (hooks) is actively listening.
        # We explicitly use named_parameters() to bypass TE attribute hiding.
        if "mlp" in name and "weight" in name and "layer_norm" not in name:
            p.requires_grad = True
            params_for_optimizer.append(p)
        else:
            # Hard-fault any attempt to touch unvirtualized global memory
            # (Attention, Embeddings, Biases, LayerNorms)
            p.requires_grad = False
    # ---------------------------------------------------------

    # Boot the Guest OS with the provisioned pointers
    opt    = torch.optim.AdamW(params_for_optimizer, lr=LR)
    scaler = torch.amp.GradScaler()
    loader = DataLoader(TokenDataset(text, tokenizer),
                        batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    step = 0
    for batch in loader:
        if step >= TRAIN_STEPS: break
        batch = batch.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch, labels=batch.clone()).loss
        scaler.scale(loss).backward()
        
        # Here, the optimizer executes its step. It thinks it is mutating the 
        # entire 768x3072 matrix, completely unaware the hooks just translated
        # the gradient into the subspace.
        scaler.step(opt)
        scaler.update()
        
        if step % 50 == 0:
            print(f"  step {step:>4}/{TRAIN_STEPS}  loss={loss.item():.4f}")
        step += 1

    freeze_domain(model, 1)
    print("  Domain 1 training complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"FP8:    {'enabled' if USE_FP8 and HAS_TE else 'disabled'}")

    print("\nLoading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", torch_dtype=torch.bfloat16).to(DEVICE)

    print("Loading corpora...")
    wt2_test    = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1",
                                         split="test")["text"])[:100_000]
    wt103_train = "\n".join(load_dataset("wikitext", "wikitext-103-raw-v1",
                                         split="train")["text"])[:200_000]

    print("\nBaseline PPL (unpatched)...")
    baseline_ppl = compute_perplexity(model, tokenizer, wt2_test, domain_id=None)
    print(f"  WikiText-2 PPL: {baseline_ppl:.3f}")

    model = patch_gpt2(model)
    prepare_domain(model, 0)
    set_active_domain(model, 0)
    freeze_domain(model, 0)

    print("\nPost-patch PPL (domain 0)...")
    post_patch_ppl = compute_perplexity(model, tokenizer, wt2_test, domain_id=0)
    print(f"  WikiText-2 PPL: {post_patch_ppl:.3f}")
    print(f"  Delta from baseline: {post_patch_ppl - baseline_ppl:+.3f}")

    train_domain1(model, tokenizer, wt103_train)

    print("\nFinal PPL (domain 0, after domain 1)...")
    final_ppl = compute_perplexity(model, tokenizer, wt2_test, domain_id=0)
    print(f"  WikiText-2 PPL: {final_ppl:.3f}")

    delta  = final_ppl - post_patch_ppl
    status = "PASS" if abs(delta) < 0.5 else "FAIL"

    print(f"\n══ RESULT ══════════════════════════════════")
    print(f"  Baseline PPL:       {baseline_ppl:.3f}")
    print(f"  Post-patch PPL:     {post_patch_ppl:.3f}")
    print(f"  After domain 1 PPL: {final_ppl:.3f}")
    print(f"  Retention delta:    {delta:+.3f}  [{status}]")

    np.save("gpt2_pwp_results.npy",
            np.array([baseline_ppl, post_patch_ppl, final_ppl]))
    print("Saved: gpt2_pwp_results.npy")

if __name__ == "__main__":
    main()