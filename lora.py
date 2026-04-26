import torch
import math
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

# --- HYPERPARAMETERS (Strictly matching the OSA run) ---
BATCH_SIZE = 8
LR = 1e-5
TRAIN_STEPS = 2000
SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_text_source(dataset_name: str, split: str):
    print(f"Loading {dataset_name} [{split}]...")
    dataset = load_dataset(dataset_name, "python", split=split)
    if "whole_func_string" in dataset.column_names:
        text = "\n\n".join(dataset["whole_func_string"][:50000]) # Match subset scale
    elif "text" in dataset.column_names:
        text = "\n\n".join(dataset["text"][:50000])
    else:
        raise ValueError("Missing text column")
    return text

def get_batches(tokens, batch_size, seq_len):
    step = batch_size * seq_len
    # Ensure we don't go out of bounds when grabbing the y_chunk (+1 offset)
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

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Load Base Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    
    # Evaluate Base English (Zero-Shot)
    print("\nLoading Base Eval Source (Wikitext)...")
    base_eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    base_eval_text = "\n\n".join(base_eval_dataset["text"])
    base_eval_tokens = tokenizer.encode(base_eval_text)
    base_ppl_before = evaluate(model, base_eval_tokens)
    print(f"Base Eval PPL (Pre-LoRA): {base_ppl_before:.3f}")

    # 2. Inject LoRA Adapter
    # Targeting the MLP modules (c_fc, c_proj) just like our PWP logic
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_fc", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 3. Load Python Data
    train_text = load_text_source("code_search_net", "train")
    eval_text = load_text_source("code_search_net", "validation")
    train_tokens = tokenizer.encode(train_text)
    eval_tokens = tokenizer.encode(eval_text)

    code_ppl_before = evaluate(model, eval_tokens)
    print(f"Python PPL (Pre-Train): {code_ppl_before:.3f}")

    # 4. Train LoRA
    print(f"\nTraining LoRA on Python ({TRAIN_STEPS} steps)...")
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

    # 5. Final Evaluations
    print("\nTraining Complete. Running Final Evaluations...")
    code_ppl_after = evaluate(model, eval_tokens)
    base_ppl_after = evaluate(model, base_eval_tokens)
    
    print("\n== LORA RESULTS ===============================")
    print(f" Trainable Parameters Added: {model.get_nb_trainable_parameters()[0]:,}")
    print(f" Base English PPL (Before):  {base_ppl_before:.3f}")
    print(f" Base English PPL (After):   {base_ppl_after:.3f}")
    print(f" Retention Delta:            {base_ppl_after - base_ppl_before:+.3f}")
    print(f" Python Code PPL (Before):   {code_ppl_before:.3f}")
    print(f" Python Code PPL (After):    {code_ppl_after:.3f}")
    print("===============================================")

if __name__ == "__main__":
    main()