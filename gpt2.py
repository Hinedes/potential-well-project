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
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
TRAIN_STEPS = 500 
EVAL_TOKENS = 8192
SEED = 42

EVAL_TEXT_FILE: Optional[str] = None
TRAIN_TEXT_FILE: Optional[str] = None
DOMAIN1_EVAL_TEXT_FILE: Optional[str] = None

EVAL_DATASET = ("wikitext", "wikitext-2-raw-v1", "test", 100_000)
TRAIN_DATASET = ("wikitext", "wikitext-103-raw-v1", "train", 200_000)
DOMAIN1_EVAL_DATASET = ("wikitext", "wikitext-103-raw-v1", "validation", 100_000)
DOMAIN1_SOURCE_MODE = "pwp_local"
DOMAIN1_SPLIT_FILE = "PWP.md"
DOMAIN1_SPLIT_TRAIN_FRACTION = 0.9
DOMAIN1_SPLIT_CHAR_LIMIT: Optional[int] = None

RUN_GENERATION_SAMPLES = True
SAMPLE_MAX_NEW_TOKENS = 64
DEFAULT_SAMPLE_PROMPTS = [
    "The future of machine learning is",
    "In a quiet research lab, the model began to",
]
PWP_SAMPLE_PROMPTS = [
    "The core insight of the Potential Well Project is",
    "In 3D weight space, multiple domains can",
]

FORCE_MODE: Optional[str] = None
TRANSITION_DEFAULT = "physical"
ALLOW_TRANSFORMER_GRASSMANNIAN = True

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


def build_importance_masks(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    n_domains: int,
):
    intermediate_size = fc1_weight.shape[0]
    if intermediate_size % n_domains != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by n_domains={n_domains}"
        )

    bucket = intermediate_size // n_domains
    with torch.no_grad():
        fc1_score = fc1_weight.float().norm(dim=1)
        fc2_score = fc2_weight.float().norm(dim=0)
        scores = fc1_score + fc2_score
        order = torch.argsort(scores, descending=True)

    masks = []
    for domain_id in range(n_domains):
        start = domain_id * bucket
        stop = start + bucket
        idx = order[start:stop]
        mask = torch.zeros(intermediate_size, dtype=torch.float32)
        mask[idx] = 1.0
        masks.append(mask)
    return masks


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
        token_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            verbose=False,
        ).input_ids[0]
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


def split_text_by_paragraph(
    text: str,
    train_fraction: float = DOMAIN1_SPLIT_TRAIN_FRACTION,
):
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if len(paragraphs) < 2:
        midpoint = max(1, min(len(text) - 1, int(len(text) * train_fraction)))
        return text[:midpoint], text[midpoint:]

    total_chars = sum(len(paragraph) + 2 for paragraph in paragraphs)
    target_chars = max(1, int(total_chars * train_fraction))

    running = 0
    split_idx = 1
    for idx, paragraph in enumerate(paragraphs[:-1], start=1):
        running += len(paragraph) + 2
        split_idx = idx
        if running >= target_chars:
            break

    split_idx = max(1, min(len(paragraphs) - 1, split_idx))
    train_text = "\n\n".join(paragraphs[:split_idx])
    eval_text = "\n\n".join(paragraphs[split_idx:])
    return train_text, eval_text


def load_domain1_train_eval_texts():
    if TRAIN_TEXT_FILE is not None or DOMAIN1_EVAL_TEXT_FILE is not None:
        train_text = load_text_source(
            text_file=TRAIN_TEXT_FILE,
            dataset_name=TRAIN_DATASET[0],
            dataset_config=TRAIN_DATASET[1],
            split=TRAIN_DATASET[2],
            char_limit=TRAIN_DATASET[3],
        )
        eval_text = load_text_source(
            text_file=DOMAIN1_EVAL_TEXT_FILE,
            dataset_name=DOMAIN1_EVAL_DATASET[0],
            dataset_config=DOMAIN1_EVAL_DATASET[1],
            split=DOMAIN1_EVAL_DATASET[2],
            char_limit=DOMAIN1_EVAL_DATASET[3],
        )
        return train_text, eval_text, {
            "source_mode": "file_or_dataset_override",
            "train_source": TRAIN_TEXT_FILE or f"{TRAIN_DATASET[0]}/{TRAIN_DATASET[1]}:{TRAIN_DATASET[2]}",
            "eval_source": DOMAIN1_EVAL_TEXT_FILE or f"{DOMAIN1_EVAL_DATASET[0]}/{DOMAIN1_EVAL_DATASET[1]}:{DOMAIN1_EVAL_DATASET[2]}",
            "eval_corpus_name": "domain1_eval",
        }

    if DOMAIN1_SOURCE_MODE == "pwp_local":
        split_path = Path(DOMAIN1_SPLIT_FILE)
        raw_text = split_path.read_text(encoding="utf-8")
        if DOMAIN1_SPLIT_CHAR_LIMIT:
            raw_text = raw_text[:DOMAIN1_SPLIT_CHAR_LIMIT]
        if not raw_text.strip():
            raise ValueError(f"{DOMAIN1_SPLIT_FILE} is empty after trimming.")

        train_text, eval_text = split_text_by_paragraph(
            raw_text,
            train_fraction=DOMAIN1_SPLIT_TRAIN_FRACTION,
        )
        return train_text, eval_text, {
            "source_mode": "pwp_local",
            "train_source": f"{DOMAIN1_SPLIT_FILE} [train split]",
            "eval_source": f"{DOMAIN1_SPLIT_FILE} [eval split]",
            "eval_corpus_name": "pwp_eval",
        }

    train_text = load_text_source(
        text_file=None,
        dataset_name=TRAIN_DATASET[0],
        dataset_config=TRAIN_DATASET[1],
        split=TRAIN_DATASET[2],
        char_limit=TRAIN_DATASET[3],
    )
    eval_text = load_text_source(
        text_file=None,
        dataset_name=DOMAIN1_EVAL_DATASET[0],
        dataset_config=DOMAIN1_EVAL_DATASET[1],
        split=DOMAIN1_EVAL_DATASET[2],
        char_limit=DOMAIN1_EVAL_DATASET[3],
    )
    return train_text, eval_text, {
        "source_mode": "dataset",
        "train_source": f"{TRAIN_DATASET[0]}/{TRAIN_DATASET[1]}:{TRAIN_DATASET[2]}",
        "eval_source": f"{DOMAIN1_EVAL_DATASET[0]}/{DOMAIN1_EVAL_DATASET[1]}:{DOMAIN1_EVAL_DATASET[2]}",
        "eval_corpus_name": "domain1_eval",
    }


def select_sample_prompts(domain1_source_mode: str):
    if domain1_source_mode == "pwp_local":
        return PWP_SAMPLE_PROMPTS
    return DEFAULT_SAMPLE_PROMPTS


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
        mid_states: Optional[list[torch.Tensor]] = None,
    ):
        super().__init__()

        if intermediate_size % n_domains != 0:
            raise ValueError("intermediate_size must be divisible by n_domains")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_domains = n_domains
        self.mode = mode
        self.k_mid = intermediate_size // n_domains
        self.layer_norm_eps = layer_norm_eps

        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = ACT2FN[activation_name]
        self.dropout = nn.Dropout(resid_pdrop)

        self.register_buffer("base_ln_weight", torch.empty(hidden_size, dtype=torch.float32))
        self.register_buffer("base_ln_bias", torch.empty(hidden_size, dtype=torch.float32))
        self.register_buffer(
            "base_fc1_weight",
            torch.empty(intermediate_size, hidden_size, dtype=torch.float32),
        )
        self.register_buffer("base_fc1_bias", torch.empty(intermediate_size, dtype=torch.float32))
        self.register_buffer(
            "base_fc2_weight",
            torch.empty(hidden_size, intermediate_size, dtype=torch.float32),
        )
        self.register_buffer("base_fc2_bias", torch.empty(hidden_size, dtype=torch.float32))

        self.frozen: set[int] = set()
        self._active = 0
        self._hooks = []

        self._register_mid_states(intermediate_size, seed + 23, mid_states)

    def _register_mid_states(
        self,
        dim: int,
        seed: int,
        mid_states: Optional[list[torch.Tensor]],
    ):
        if mid_states is not None:
            if len(mid_states) != self.n_domains:
                raise ValueError("mid_states length must match n_domains")
            for domain_id, state in enumerate(mid_states):
                self.register_buffer(f"mid_state_{domain_id}", state.clone().float())
            return

        if self.mode == "grassmannian":
            bases = round_robin_bases(dim, self.n_domains, self.k_mid, seed)
            for domain_id, basis in enumerate(bases):
                self.register_buffer(f"mid_state_{domain_id}", basis)
            return

        for domain_id in range(self.n_domains):
            mask = torch.zeros(dim, dtype=torch.float32)
            start = domain_id * self.k_mid
            stop = start + self.k_mid
            mask[start:stop] = 1.0
            self.register_buffer(f"mid_state_{domain_id}", mask)

    def _mid_state(self, domain_id: int):
        return getattr(self, f"mid_state_{domain_id}")

    def capture_base_state(self):
        with torch.no_grad():
            self.base_ln_weight.copy_(self.ln.weight.detach().float())
            self.base_ln_bias.copy_(self.ln.bias.detach().float())
            self.base_fc1_weight.copy_(self.fc1.weight.detach().float())
            self.base_fc1_bias.copy_(self.fc1.bias.detach().float())
            self.base_fc2_weight.copy_(self.fc2.weight.detach().float())
            self.base_fc2_bias.copy_(self.fc2.bias.detach().float())

    def _project_mid_features(self, tensor: torch.Tensor, domain_id: int):
        state = self._mid_state(domain_id)
        if self.mode == "grassmannian":
            basis = state.to(device=tensor.device, dtype=tensor.dtype)
            return (tensor @ basis) @ basis.T
        mask = state.to(device=tensor.device, dtype=tensor.dtype)
        return tensor * mask

    def _project_mid_vector(self, grad: torch.Tensor, domain_id: int):
        state = self._mid_state(domain_id)
        if self.mode == "grassmannian":
            basis = state.to(device=grad.device, dtype=grad.dtype)
            return basis @ (basis.T @ grad)
        mask = state.to(device=grad.device, dtype=grad.dtype)
        return grad * mask

    def _project_fc1_weight(self, grad: torch.Tensor, domain_id: int):
        row_state = self._mid_state(domain_id)
        if self.mode == "grassmannian":
            row_basis = row_state.to(device=grad.device, dtype=grad.dtype)
            return row_basis @ (row_basis.T @ grad)
        row_mask = row_state.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
        return grad * row_mask

    def _project_fc2_weight(self, grad: torch.Tensor, domain_id: int):
        col_state = self._mid_state(domain_id)
        if self.mode == "grassmannian":
            col_basis = col_state.to(device=grad.device, dtype=grad.dtype)
            return (grad @ col_basis) @ col_basis.T
        col_mask = col_state.to(device=grad.device, dtype=grad.dtype).unsqueeze(0)
        return grad * col_mask

    def domain_parameters(self):
        yield from self.ln.parameters()
        yield from self.fc1.parameters()
        yield from self.fc2.parameters()

    def prepare_domain(self, domain_id: int):
        if self.mode != "grassmannian":
            return

        with torch.no_grad():
            frozen_bases = [
                self._mid_state(frozen_id).detach()
                for frozen_id in sorted(self.frozen)
            ]
            if not frozen_bases:
                return
            updated = orthogonalize_against_frozen(
                self._mid_state(domain_id), frozen_bases
            )
            self._mid_state(domain_id).copy_(updated)

    def freeze_domain(self, domain_id: int):
        self.frozen.add(domain_id)

    def set_active(self, domain_id: int):
        self._active = domain_id
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        if domain_id == 0:
            return

        self._hooks.append(
            self.fc1.weight.register_hook(
                lambda grad, d=domain_id: self._project_fc1_weight(grad, d)
            )
        )
        self._hooks.append(
            self.fc1.bias.register_hook(
                lambda grad, d=domain_id: self._project_mid_vector(grad, d)
            )
        )
        self._hooks.append(
            self.fc2.weight.register_hook(
                lambda grad, d=domain_id: self._project_fc2_weight(grad, d)
            )
        )

    def _forward_base(self, x: torch.Tensor):
        h = F.layer_norm(
            x,
            (self.hidden_size,),
            weight=self.base_ln_weight.to(device=x.device, dtype=x.dtype),
            bias=self.base_ln_bias.to(device=x.device, dtype=x.dtype),
            eps=self.layer_norm_eps,
        )
        h = F.linear(
            h,
            self.base_fc1_weight.to(device=x.device, dtype=x.dtype),
            self.base_fc1_bias.to(device=x.device, dtype=x.dtype),
        )
        h = self.act(h)
        out = F.linear(
            h,
            self.base_fc2_weight.to(device=x.device, dtype=x.dtype),
            self.base_fc2_bias.to(device=x.device, dtype=x.dtype),
        )
        return self.dropout(out)

    def forward(self, x: torch.Tensor):
        if self._active == 0:
            return self._forward_base(x)

        x = self.ln(x)
        h = self.fc1(x)
        h = self._project_mid_features(h, self._active)
        h = self.act(h)
        h = self._project_mid_features(h, self._active)
        out = self.fc2(h)
        return self.dropout(out)


def iter_pwp_blocks(model: GPT2LMHeadModel):
    for block in model.transformer.h:
        if isinstance(block.mlp, PWPMLPBlock):
            yield block.mlp


def patch_gpt2(model: GPT2LMHeadModel):
    hidden_size = model.config.n_embd
    intermediate_size = model.config.n_inner or (4 * hidden_size)
    threshold_mode, k, reason = select_architecture(
        intermediate_size, N_DOMAINS, FORCE_MODE
    )
    mode = threshold_mode
    effective_reason = reason

    if (
        threshold_mode == "grassmannian"
        and FORCE_MODE is None
        and not ALLOW_TRANSFORMER_GRASSMANNIAN
    ):
        mode = "physical"
        effective_reason = (
            f"threshold chose grassmannian ({reason}), but GPT-2 falls back to "
            "physical FFN isolation to avoid destroying the pretrained residual stream"
        )

    print(
        f"\nPatching GPT-2 ({len(model.transformer.h)} layers) "
        f"with mode={mode}, k={k} [{effective_reason}]..."
    )

    for layer_idx, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, PWPMLPBlock):
            continue

        source_mlp = block.mlp
        source_ln = block.ln_2
        mid_states = None
        if mode == "physical":
            mid_states = build_importance_masks(
                source_mlp.c_fc.weight.t(),
                source_mlp.c_proj.weight.t(),
                N_DOMAINS,
            )

        pwp = PWPMLPBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_domains=N_DOMAINS,
            activation_name=model.config.activation_function,
            layer_norm_eps=model.config.layer_norm_epsilon,
            resid_pdrop=model.config.resid_pdrop,
            mode=mode,
            seed=SEED + layer_idx * 101,
            mid_states=mid_states,
        ).to(device=source_ln.weight.device)

        with torch.no_grad():
            pwp.ln.weight.copy_(source_ln.weight.float())
            pwp.ln.bias.copy_(source_ln.bias.float())
            pwp.fc1.weight.copy_(source_mlp.c_fc.weight.t().float())
            pwp.fc1.bias.copy_(source_mlp.c_fc.bias.float())
            pwp.fc2.weight.copy_(source_mlp.c_proj.weight.t().float())
            pwp.fc2.bias.copy_(source_mlp.c_proj.bias.float())
            pwp.capture_base_state()

        block.mlp = pwp
        block.ln_2 = nn.Identity()

    return model, mode, k, effective_reason


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

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        verbose=False,
    ).input_ids[0]
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


def compute_route_ppl_matrix(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    corpora: list[tuple[str, str]],
    domain_ids: list[int],
):
    matrix = {}
    for corpus_name, corpus_text in corpora:
        row = {}
        for domain_id in domain_ids:
            row[f"domain_{domain_id}"] = compute_perplexity(
                model,
                tokenizer,
                corpus_text,
                domain_id=domain_id,
            )
        matrix[corpus_name] = row
    return matrix


def print_route_ppl_matrix(title: str, matrix: dict[str, dict[str, float]]):
    print(f"\n== {title} ===================")
    if not matrix:
        print("  <empty>")
        return

    row_names = list(matrix.keys())
    col_names = list(next(iter(matrix.values())).keys())

    header = f"{'Corpus':<18}" + "".join(f"{col:<16}" for col in col_names)
    print(header)
    print("-" * len(header))
    for row_name in row_names:
        row = matrix[row_name]
        values = "".join(f"{row[col]:<16.3f}" for col in col_names)
        print(f"{row_name:<18}{values}")


def compute_route_margins(
    matrix: dict[str, dict[str, float]],
    *,
    base_corpus_name: str,
    domain_corpus_name: str,
    train_domain: int,
):
    train_key = f"domain_{train_domain}"
    return {
        "base_corpus_prefers_domain_0_by": (
            matrix[base_corpus_name][train_key] - matrix[base_corpus_name]["domain_0"]
        ),
        "domain_corpus_prefers_train_domain_by": (
            matrix[domain_corpus_name]["domain_0"] - matrix[domain_corpus_name][train_key]
        ),
    }


def print_route_margins(title: str, margins: dict[str, float]):
    print(f"\n== {title} ====================")
    for key, value in margins.items():
        print(f"  {key}: {value:+.3f}")


@torch.no_grad()
def generate_sample(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    *,
    domain_id: int,
    max_new_tokens: int = SAMPLE_MAX_NEW_TOKENS,
    seed_offset: int = 0,
):
    model.eval()
    set_active_domain(model, domain_id)

    previous_use_cache = model.config.use_cache
    model.config.use_cache = True

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, model.config.n_positions - max_new_tokens),
        verbose=False,
    ).to(DEVICE)

    seed_value = SEED + domain_id * 100 + seed_offset
    fork_devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed_value)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed_value)

        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=0.6,   # Lowered from 0.8 to reduce gibberish
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        
    model.config.use_cache = previous_use_cache
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
    train_text, domain1_eval_text, domain1_meta = load_domain1_train_eval_texts()
    domain1_eval_name = domain1_meta["eval_corpus_name"]
    sample_prompts = select_sample_prompts(domain1_meta["source_mode"])

    print(f"  Base eval source:    {EVAL_TEXT_FILE or f'{EVAL_DATASET[0]}/{EVAL_DATASET[1]}:{EVAL_DATASET[2]}'}")
    print(f"  Domain {TRAIN_DOMAIN} train:  {domain1_meta['train_source']}")
    print(f"  Domain {TRAIN_DOMAIN} eval:   {domain1_meta['eval_source']}")

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

    print(f"\nDomain {TRAIN_DOMAIN} PPL before training (held-out domain text)...")
    domain1_pre_ppl = compute_perplexity(
        model,
        tokenizer,
        domain1_eval_text,
        domain_id=TRAIN_DOMAIN,
    )
    print(f"  Held-out PPL: {domain1_pre_ppl:.3f}")

    pre_train_route_matrix = compute_route_ppl_matrix(
        model,
        tokenizer,
        [
            ("base_eval", eval_text),
            (domain1_eval_name, domain1_eval_text),
        ],
        [0, TRAIN_DOMAIN],
    )
    print_route_ppl_matrix("PRE-TRAIN ROUTE MATRIX", pre_train_route_matrix)
    pre_train_route_margins = compute_route_margins(
        pre_train_route_matrix,
        base_corpus_name="base_eval",
        domain_corpus_name=domain1_eval_name,
        train_domain=TRAIN_DOMAIN,
    )
    print_route_margins("PRE-TRAIN ROUTE MARGINS", pre_train_route_margins)

    train_domain(model, tokenizer, train_text, domain_id=TRAIN_DOMAIN)

    print("\nFinal PPL (domain 0, after domain 1)...")
    final_ppl = compute_perplexity(model, tokenizer, eval_text, domain_id=0)
    print(f"  Eval PPL: {final_ppl:.3f}")

    print(f"\nDomain {TRAIN_DOMAIN} PPL after training (held-out domain text)...")
    domain1_post_ppl = compute_perplexity(
        model,
        tokenizer,
        domain1_eval_text,
        domain_id=TRAIN_DOMAIN,
    )
    print(f"  Held-out PPL: {domain1_post_ppl:.3f}")

    post_train_route_matrix = compute_route_ppl_matrix(
        model,
        tokenizer,
        [
            ("base_eval", eval_text),
            (domain1_eval_name, domain1_eval_text),
        ],
        [0, TRAIN_DOMAIN],
    )
    print_route_ppl_matrix("POST-TRAIN ROUTE MATRIX", post_train_route_matrix)
    post_train_route_margins = compute_route_margins(
        post_train_route_matrix,
        base_corpus_name="base_eval",
        domain_corpus_name=domain1_eval_name,
        train_domain=TRAIN_DOMAIN,
    )
    print_route_margins("POST-TRAIN ROUTE MARGINS", post_train_route_margins)

    delta = final_ppl - post_patch_ppl
    domain1_delta = domain1_post_ppl - domain1_pre_ppl
    status = "PASS" if abs(delta) < 0.5 else "FAIL"

    print("\n== RESULT ===============================")
    print(f"  Baseline PPL:       {baseline_ppl:.3f}")
    print(f"  Post-patch PPL:     {post_patch_ppl:.3f}")
    print(f"  After domain 1 PPL: {final_ppl:.3f}")
    print(f"  Retention delta:    {delta:+.3f}  [{status}]")
    print(f"  Domain {TRAIN_DOMAIN} pre PPL:  {domain1_pre_ppl:.3f}")
    print(f"  Domain {TRAIN_DOMAIN} post PPL: {domain1_post_ppl:.3f}")
    print(f"  Domain {TRAIN_DOMAIN} delta:    {domain1_delta:+.3f}")

    generation_samples = []
    if RUN_GENERATION_SAMPLES:
        print("\n== GENERATION SAMPLES ===================")
        for sample_idx, prompt in enumerate(sample_prompts):
            try:
                domain0_text = generate_sample(
                    model,
                    tokenizer,
                    prompt,
                    domain_id=0,
                    seed_offset=sample_idx,
                )
                domain1_text = generate_sample(
                    model,
                    tokenizer,
                    prompt,
                    domain_id=TRAIN_DOMAIN,
                    seed_offset=sample_idx,
                )
                generation_samples.append(
                    {
                        "prompt": prompt,
                        "domain_0": domain0_text,
                        f"domain_{TRAIN_DOMAIN}": domain1_text,
                    }
                )
                print(f"\nPrompt: {prompt}")
                print(f"  Domain 0: {domain0_text}")
                print(f"  Domain {TRAIN_DOMAIN}: {domain1_text}")
            except Exception as exc:
                generation_samples.append(
                    {
                        "prompt": prompt,
                        "error": str(exc),
                    }
                )
                print(f"\nPrompt: {prompt}")
                print(f"  Generation failed: {exc}")

    np.save(
        "gpt2_pwp_results.npy",
        np.array([baseline_ppl, post_patch_ppl, final_ppl], dtype=np.float32),
    )
    print("Saved: gpt2_pwp_results.npy")

    report = {
        "model_name": MODEL_NAME,
        "n_domains": N_DOMAINS,
        "train_domain": TRAIN_DOMAIN,
        "mode": mode,
        "k": k,
        "mode_reason": reason,
        "base_eval_source": EVAL_TEXT_FILE or f"{EVAL_DATASET[0]}/{EVAL_DATASET[1]}:{EVAL_DATASET[2]}",
        "domain1_source_mode": domain1_meta["source_mode"],
        "domain1_train_source": domain1_meta["train_source"],
        "domain1_eval_source": domain1_meta["eval_source"],
        "domain1_eval_corpus_name": domain1_eval_name,
        "baseline_ppl": baseline_ppl,
        "post_patch_domain0_ppl": post_patch_ppl,
        "final_domain0_ppl": final_ppl,
        "domain0_retention_delta": delta,
        "domain0_status": status,
        "domain1_pre_ppl": domain1_pre_ppl,
        "domain1_post_ppl": domain1_post_ppl,
        "domain1_delta": domain1_delta,
        "pre_train_route_matrix": pre_train_route_matrix,
        "pre_train_route_margins": pre_train_route_margins,
        "post_train_route_matrix": post_train_route_matrix,
        "post_train_route_margins": post_train_route_margins,
        "sample_prompts": sample_prompts,
        "generation_samples": generation_samples,
    }
    Path("gpt2_pwp_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("Saved: gpt2_pwp_results.json")


if __name__ == "__main__":
    main()
