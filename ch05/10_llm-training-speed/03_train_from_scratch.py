# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch.utils.tensorboard import SummaryWriter

import fineweb_dataloader
from dotenv import load_dotenv
from pathlib import Path
import math
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


#####################################
# Chapter 3
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Chapter 5
#####################################


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calculate_learning_rate(global_step, total_global_step):
    initial_lr = 0.0001
    peak_lr = 0.005
    min_lr = 0.1 * initial_lr

    warmup_steps = int(0.2 * total_global_step)
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    if global_step < warmup_steps:
        lr = initial_lr + global_step * lr_increment  
    else:
        progress = ((global_step - warmup_steps) / (total_global_step - warmup_steps))
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr

def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    model.train()
    return decoded_text


def train_model_simple_with_timing(model, train_loader, train_loader_fixed, val_loader_fixed, optimizer, device,
                                   num_epochs, global_total_step, start_context, tokenizer, grad_accum_steps):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, last_tokens = 0, 0
    scaler = GradScaler()

    micro_step = 0
    global_step = 0
    # Variables for cumulative average tokens/sec
    cumulative_time = 0.0

    # CUDA-specific timing setup
    use_cuda = device.type == "cuda"
    log_writer = SummaryWriter(flush_secs=5)

    validation_step   = max(1, int(global_total_step * 0.01))
    print_sample_step = max(1, int(global_total_step * 0.02))
    save_model_step   = max(1, int(global_total_step * 0.10))

    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
        t_start.record()          # Start the timer for the first interval
    else:
        t0 = time.time()          # Start the timer for the first interval
    optimizer.zero_grad(set_to_none=True)
    

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        for inp_batch, tgt_batch in train_loader:
            micro_step += 1
            total_tokens += inp_batch.numel()

            # Forward and backward pass
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            scaler.scale(loss / grad_accum_steps).backward()

            if (micro_step % grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                grad_norm_pre = clip_grad_norm_(model.parameters(), max_norm=float("inf"), norm_type=2).item()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norm_post = clip_grad_norm_(model.parameters(), max_norm=float("inf"), norm_type=2).item()

                log_writer.add_scalar("gradident/norm_pre_clip",  grad_norm_pre,  global_step=global_step)
                log_writer.add_scalar("gradident/norm_post_clip", grad_norm_post, global_step=global_step)


                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                lr = calculate_learning_rate(global_step, global_total_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr   


                # At evaluation intervals, measure elapsed time and tokens per second
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Reset timer for the next interval

                # Calculate tokens processed in this interval
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                cumulative_time += elapsed

                # Compute cumulative average tokens/sec (excluding the first interval)
                avg_tps = float(total_tokens) / cumulative_time if cumulative_time > 0 else 0

                log_writer.add_scalar("learning_rate", lr, global_step=global_step)
                log_writer.add_scalar("Loss/train_model", float(loss.item()), global_step=global_step)

                log_writer.add_scalar("token/total", total_tokens, global_step=global_step)
                log_writer.add_scalar("token/interval", tokens_interval, global_step=global_step)

                log_writer.add_scalar("time/elapsed", elapsed, global_step=global_step)
                log_writer.add_scalar("time/cumulative_time", cumulative_time, global_step=global_step)

                log_writer.add_scalar("TPS/average", avg_tps, global_step=global_step)
                log_writer.add_scalar("TPS/per_second", round(tps), global_step=global_step)

                print(f"Ep {epoch+1}, Step {global_step:06d}, Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}. TrainLoss {round(loss.item())}")


                # logging and testing
                if global_step % validation_step == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader_fixed, val_loader_fixed, device)
                    log_writer.add_scalar("Loss_eval/train", train_loss, global_step=global_step)
                    log_writer.add_scalar("Loss_eval/validation", val_loss, global_step=global_step)

                    if torch.cuda.is_available():
                        device = torch.cuda.current_device()

                        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
                        reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

                        log_writer.add_scalar("memory/allocatedGB", allocated, global_step=global_step)
                        log_writer.add_scalar("memory/reservedGB", reserved, global_step=global_step)


                if global_step % print_sample_step == 0:
                    result = generate_and_print_sample(model, tokenizer, device, start_context)
                    log_writer.add_text("sample_response", result, global_step=global_step)

                if global_step % save_model_step == 0:
                    save_model(model, optimizer, scaler, global_step, epoch, micro_step, total_tokens, "checkpoints", f"step{global_step:06d}")

    return train_losses, val_losses, track_tokens


#####################################
# Main function calls
#####################################

def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"Uses tensor cores: {torch.cuda.is_available()}")
    print()


    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)  
    model = torch.compile(model)
    model.to(device)

    lr = calculate_learning_rate(0, settings["global_total_step"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=settings["weight_decay"],
        fused=True
    )

    train_loader, val_loader = fineweb_dataloader.get_fineweb_loaders(
        max_length=gpt_config["context_length"], batch_size=settings["batch_size"], num_workers=4, add_eot=True
    )

    train_eval_loader, val_eval_loader = fineweb_dataloader.make_fixed_eval_loaders(
        train_loader, val_loader, max_train_batches=8, max_val_batches=16
    )
 

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        train_loader_fixed=train_eval_loader,
        val_loader_fixed=val_eval_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        global_total_step=settings["global_total_step"],
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        grad_accum_steps=settings["grad_accum_steps"]
    )

    return train_losses, val_losses, tokens_seen, model


def save_model(model, optimizer, scaler, global_step, epoch, micro_step, total_tokens, folder, filename):
    compiled = hasattr(model, "_orig_mod")
    os.makedirs(folder, exist_ok=True) 
    filepath = os.path.join(folder, f"{filename}-checkpoint.pth")
    
    checkpoint = {
        'model_state_dict': model._orig_mod.state_dict() if compiled else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'micro_step': micro_step,
        'total_tokens': total_tokens,
        'config': None  # You can add your GPT_CONFIG_124M here if needed
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

if __name__ == "__main__":
    load_dotenv(dotenv_path=Path("/teamspace/studios/this_studio/LLMs-from-scratch/.env")) 
    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # Vocabulary size
        "context_length": 1024,  # Input tokens per training example
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.05,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "num_epochs": 1,
        "batch_size": 64,
        "weight_decay": 0.1,
        "global_total_step" : 30000,
        "grad_accum_steps": 2
    }

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
