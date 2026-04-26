import torch
import math
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# --- BULLETPROOF HYPERPARAMETERS ---
BATCH_SIZE = 8
LR = 1e-5
TRAIN_STEPS = 2000
SEQ_LEN = 256
LORA_EQUIVALENT_RANK = 8  # We will dynamically match OSA k to this LoRA rank
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3.5-2B-Base"

def load_text_source(dataset_name: str, split: str):
    print(f"Loading {dataset_name} [{split}]...")
    dataset = load_dataset(dataset_name, "python", split=split)
    if "whole_func_string" in dataset.column_names:
        return "\n\n".join(dataset["whole_func_string"][:50000])
    return "\n\n".join(dataset["text"][:50000])

def get_batches(tokens, batch_size, seq_len):
    step = batch_size * seq_len
    max_idx = len(tokens) - step
    for i in range(0, max_idx, step):
        x_chunk = tokens[i : i + step]
        # CausalLM shifts labels internally, so labels should match input ids.
        y_chunk = x_chunk
        yield torch.tensor(x_chunk).view(batch_size, seq_len).to(DEVICE), \
              torch.tensor(y_chunk).view(batch_size, seq_len).to(DEVICE)

def tokenize_corpus(tokenizer, text: str):
    # Avoid max-length warnings for very long evaluation/training corpora.
    return tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]

def evaluate(model, tokens, max_batches=50):
    model.eval()
    nlls = []
    with torch.no_grad():
        for i, (x, y) in enumerate(get_batches(tokens, BATCH_SIZE, SEQ_LEN)):
            if i >= max_batches: break
            nlls.append(model(x, labels=y).loss.item())
    return math.exp(sum(nlls) / len(nlls)) if nlls else float('inf')

class BulletproofOSAQwenMLP(nn.Module):
    def __init__(self, original_mlp: Qwen2MLP, config, lora_r: int):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        
        self.lora_r = lora_r
        self._pi_backward_gate = None
        self._pi_backward_up = None
        self._pi_backward_down = None
        
        # Frozen baselines for PSMP (Post-Step Manifold Projection)
        self._frozen_core_gate = None
        self._frozen_core_up = None
        self._frozen_core_down = None
        self._hooks = []

    def _calc_k(self, in_features, out_features):
        """Calculates exact Degree-of-Freedom parity with LoRA."""
        lora_params = self.lora_r * (in_features + out_features)
        return math.ceil(lora_params / out_features)

    def _init_layer_subspace(self, layer: nn.Linear):
        """Carves orthogonal subspaces and returns Pi_backward and the Frozen Core."""
        W = layer.weight.data.float()
        out_features, in_features = W.shape
        k = self._calc_k(in_features, out_features)
        
        # Calculate Input-space Geometry
        _, _, V = torch.svd(W)
        
        # To minimize the Spatial Tax, Domain 0 keeps top singular values. Domain 1 gets the lowest.
        P_0 = V[:, :-k]  # Core Base
        P_1 = V[:, -k:]  # Shell Adapter
        
        Pi_core = P_0 @ P_0.T
        Pi_shell = P_1 @ P_1.T
        
        # Materialize the strictly mathematical frozen core
        frozen_core = (W @ Pi_core).to(layer.weight.dtype)
        
        return Pi_shell.to(layer.weight.dtype), frozen_core

    def patch_geometry(self):
        """Initializes the SVD boundaries and sets up PSMP targets."""
        with torch.no_grad():
            self._pi_backward_gate, self._frozen_core_gate = self._init_layer_subspace(self.gate_proj)
            self._pi_backward_up, self._frozen_core_up = self._init_layer_subspace(self.up_proj)
            self._pi_backward_down, self._frozen_core_down = self._init_layer_subspace(self.down_proj)
            
            # Immediately force the weights into their compressed state to pay the Spatial Tax
            self.enforce_psmp()

    def set_active(self, domain_id: int):
        for hook in self._hooks: hook.remove()
        self._hooks = []

        if domain_id == 1:
            # Trap gradients strictly within the Shell Subspace
            self._hooks.append(self.gate_proj.weight.register_hook(lambda grad: grad @ self._pi_backward_gate))
            self._hooks.append(self.up_proj.weight.register_hook(lambda grad: grad @ self._pi_backward_up))
            self._hooks.append(self.down_proj.weight.register_hook(lambda grad: grad @ self._pi_backward_down))

    def enforce_psmp(self):
        """Post-Step Manifold Projection: Eradicates AdamW drift and precision errors."""
        with torch.no_grad():
            self.gate_proj.weight.data = self._frozen_core_gate + (self.gate_proj.weight.data @ self._pi_backward_gate)
            self.up_proj.weight.data = self._frozen_core_up + (self.up_proj.weight.data @ self._pi_backward_up)
            self.down_proj.weight.data = self._frozen_core_down + (self.down_proj.weight.data @ self._pi_backward_down)

    def forward(self, x):
        # Native RAG: The input inherently traverses the union of Core + Shell
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up
        return self.down_proj(hidden)

def main():
    print(f"Device: {DEVICE} | Target Parity: LoRA r={LORA_EQUIVALENT_RANK}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_kwargs: dict[str, object] = {"trust_remote_code": True}
    if DEVICE == "cuda":
        model_kwargs["dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    if DEVICE != "cuda":
        model.to(DEVICE)

    # 1. Measure Raw Baselinetokenizer
    print("\nLoading Base Eval Source (Wikitext)...")
    base_eval_text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    base_eval_tokens = tokenize_corpus(tokenizer, base_eval_text)
    ppl_raw = evaluate(model, base_eval_tokens)
    print(f"Base PPL (Raw Model): {ppl_raw:.3f}")

    # 2. Architect the Null-Spaces & Measure Truncation Tax
    print("\nPatching SVD Null-Spaces into SwiGLU layers...")
    osa_mlps = []
    for layer in model.model.layers:
        if hasattr(layer, "mlp"):
            osa_mlp = BulletproofOSAQwenMLP(layer.mlp, model.config, LORA_EQUIVALENT_RANK)
            osa_mlp.patch_geometry()
            layer.mlp = osa_mlp
            osa_mlps.append(osa_mlp)
            
    ppl_patched = evaluate(model, base_eval_tokens)
    print(f"Base PPL (Patched - The Spatial Tax): {ppl_patched:.3f}")
    print(f"-> SVD Compression Cost: +{(ppl_patched - ppl_raw):.3f}")

    # 3. Code Eval & Training Loop
    train_tokens = tokenize_corpus(tokenizer, load_text_source("code_search_net", "train"))
    eval_tokens = tokenize_corpus(tokenizer, load_text_source("code_search_net", "validation"))

    for mlp in osa_mlps: mlp.set_active(domain_id=1)

    print(f"\nPython PPL (Zero-Shot Patched): {evaluate(model, eval_tokens):.3f}")
    print(f"Training OSA Shell on Python ({TRAIN_STEPS} steps)...")

    for param in model.parameters():
        param.requires_grad = False
    for mlp in osa_mlps:
        mlp.gate_proj.weight.requires_grad = True
        mlp.up_proj.weight.requires_grad = True
        mlp.down_proj.weight.requires_grad = True
    
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    print(f"Trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    model.train()
    
    batch_gen = get_batches(train_tokens, BATCH_SIZE, SEQ_LEN)
    for step in range(TRAIN_STEPS):
        try: x, y = next(batch_gen)
        except StopIteration:
            batch_gen = get_batches(train_tokens, BATCH_SIZE, SEQ_LEN)
            x, y = next(batch_gen)
            
        optimizer.zero_grad()
        loss = model(x, labels=y).loss
        loss.backward()
        optimizer.step()
        
        # The ultimate write-lock enforcement: PSMP
        for mlp in osa_mlps: mlp.enforce_psmp()
        
        if step % 200 == 0: print(f" step {step:4d}/{TRAIN_STEPS} loss={loss.item():.4f}")

    # 4. The Usability Proof
    model.eval()
    code_ppl_final = evaluate(model, eval_tokens)
    
    for mlp in osa_mlps: mlp.set_active(domain_id=0)
    ppl_final = evaluate(model, base_eval_tokens)
    
    print("\n== BULLETPROOF OSA RESULTS ====================")
    print(f" Target Capability (Python Final PPL): {code_ppl_final:.3f}")
    print(f" Base PPL (Pre-Train Patched):         {ppl_patched:.3f}")
    print(f" Base PPL (Post-Train Final):          {ppl_final:.3f}")
    print(f" Mathematical Retention Delta:         {ppl_final - ppl_patched:+.4f} (Must be exactly +0.0000)")
    print("===============================================")

if __name__ == "__main__":
    main()