import torch
import math
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8
LR = 1e-5
TRAIN_STEPS = 2000
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_DIM = 1024 # The Grassmannian capacity for the OSA Shell

def load_text_source(dataset_name: str, split: str):
    print(f"Loading {dataset_name} [{split}]...")
    dataset = load_dataset(dataset_name, "python", split=split)
    if "whole_func_string" in dataset.column_names:
        text = "\n\n".join(dataset["whole_func_string"][:50000])
    else:
        text = "\n\n".join(dataset["text"][:50000])
    return text

def get_batches(tokens, batch_size, seq_len):
    step = batch_size * seq_len
    max_idx = len(tokens) - step - 1 
    
    for i in range(0, max_idx, step):
        x_chunk = tokens[i : i + step]
        y_chunk = tokens[i + 1 : i + 1 + step]
        
        x = torch.tensor(x_chunk).view(batch_size, seq_len)
        y = torch.tensor(y_chunk).view(batch_size, seq_len)
        yield x.to(DEVICE), y.to(DEVICE)

def evaluate(model, tokens, max_batches=50):
    model.eval()
    nlls = []
    with torch.no_grad():
        batches = get_batches(tokens, BATCH_SIZE, SEQ_LEN)
        for i, (x, y) in enumerate(batches):
            if i >= max_batches: break
            out = model(x, labels=y)
            nlls.append(out.loss.item())
    return math.exp(sum(nlls) / len(nlls)) if nlls else float('inf')

class OSAQwenMLP(nn.Module):
    def __init__(self, original_mlp: Qwen2MLP, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Native dense layers
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        
        self._active_domain = 0
        self._pi_forward = None
        self._pi_backward = None
        self._hooks = []
        
        # SVD Basis Cache
        self._bases = {}

    def init_svd_subspaces(self, k: int):
        """Calculates P_0 (Core) and P_1 (Shell) using SVD on the down_proj weights."""
        with torch.no_grad():
            W = self.down_proj.weight.data.float()
            U, S, V = torch.svd(W)
            
            # Domain 0 (Core) gets the principal subspace
            P_0 = V[:, :k]
            self._bases[0] = P_0
            
            # Domain 1 (Shell) gets the strict orthogonal complement
            P_1 = V[:, k:2*k]
            # Gram-Schmidt to guarantee absolute orthogonality due to float precision
            P_1 = P_1 - P_0 @ (P_0.T @ P_1)
            P_1, _ = torch.linalg.qr(P_1)
            self._bases[1] = P_1

    def set_active(self, domain_id: int):
        self._active_domain = domain_id
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        if domain_id == 0:
            self._pi_forward = None
            self._pi_backward = None
            return

        P_0 = self._bases[0].to(device=self.down_proj.weight.device, dtype=self.down_proj.weight.dtype)
        P_1 = self._bases[1].to(device=self.down_proj.weight.device, dtype=self.down_proj.weight.dtype)

        # OSA Gate Math
        pi_core = P_0 @ P_0.T
        self._pi_backward = P_1 @ P_1.T
        self._pi_forward = pi_core + self._pi_backward

        # The Write Lock: Trap gradients in the orthogonal shell
        self._hooks.append(self.gate_proj.weight.register_hook(lambda grad: grad @ self._pi_backward))
        self._hooks.append(self.up_proj.weight.register_hook(lambda grad: grad @ self._pi_backward))
        self._hooks.append(self.down_proj.weight.register_hook(lambda grad: self._pi_backward @ grad))

    def forward(self, x):
        if self._active_domain != 0 and self._pi_forward is not None:
            # Read gate: filter input through the allowed union space
            x = x @ self._pi_forward
            
        # Standard SwiGLU execution
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up
        out = self.down_proj(hidden)
        
        if self._active_domain != 0 and self._pi_forward is not None:
            # Maintain geometry on the residual stream
            out = out @ self._pi_forward
            
        return out

def patch_model(model, k_dim):
    print(f"Patching Qwen MLP blocks with OSA (k={k_dim})...")
    patched_count = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp"):
            osa_mlp = OSAQwenMLP(layer.mlp, model.config)
            osa_mlp.init_svd_subspaces(k=k_dim)
            layer.mlp = osa_mlp
            patched_count += 1
    print(f"Patched {patched_count} layers.")
    return model

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Load Base Model (Qwen 3.5 - 1.7B)
    model_id = "Qwen/Qwen3.5-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map=DEVICE, 
        trust_remote_code=True
    )
    
    # 2. Evaluate Base English (Domain 0)
    print("\nLoading Base Eval Source (Wikitext)...")
    base_eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    base_eval_text = "\n\n".join(base_eval_dataset["text"])
    base_eval_tokens = tokenizer.encode(base_eval_text)
    base_ppl_before = evaluate(model, base_eval_tokens)
    print(f"Base Eval PPL (Unpatched): {base_ppl_before:.3f}")

    # 3. Patch Architecture
    model = patch_model(model, K_DIM)

    # 4. Evaluate Python (Zero-Shot via SVD Core Read)
    train_text = load_text_source("code_search_net", "train")
    eval_text = load_text_source("code_search_net", "validation")
    train_tokens = tokenizer.encode(train_text)
    eval_tokens = tokenizer.encode(eval_text)

    # Set active to Domain 1 (Python)
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "set_active"):
            layer.mlp.set_active(domain_id=1)

    code_ppl_before = evaluate(model, eval_tokens)
    print(f"Python PPL (Pre-Train, reading Core): {code_ppl_before:.3f}")

    # 5. Train Domain 1 Shell
    print(f"\nTraining OSA Shell on Python ({TRAIN_STEPS} steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    
    step = 0
    batch_gen = get_batches(train_tokens, BATCH_SIZE, SEQ_LEN)
    
    while step < TRAIN_STEPS:
        try:
            x, y = next(batch_gen)
        except StopIteration:
            batch_gen = get_batches(train_tokens, BATCH_SIZE, SEQ_LEN)
            x, y = next(batch_gen)
            
        optimizer.zero_grad()
        loss = model(x, labels=y).loss
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f" step {step:4d}/{TRAIN_STEPS} loss={loss.item():.4f}")
        step += 1

    # 6. Final Evaluations
    print("\nTraining Complete. Running Final Evaluations...")
    model.eval()
    code_ppl_after = evaluate(model, eval_tokens)
    
    # Switch back to Domain 0 to verify Write-Lock
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "set_active"):
            layer.mlp.set_active(domain_id=0)
    base_ppl_after = evaluate(model, base_eval_tokens)
    
    print("\n== OSA RESULTS (QWEN 3.5) =====================")
    print(f" Base English PPL (Before):  {base_ppl_before:.3f}")
    print(f" Base English PPL (After):   {base_ppl_after:.3f}")
    print(f" Retention Delta:            {base_ppl_after - base_ppl_before:+.3f}")
    print(f" Python Code PPL (Before):   {code_ppl_before:.3f}")
    print(f" Python Code PPL (After):    {code_ppl_after:.3f}")
    print("===============================================")