import os
import time
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset as TorchDataset, get_worker_info

MIN_LANGUAGE_SCORE = 0.9

def is_en_high_conf(ex):
    if ex.get("language") != "en":
        return False
    if ex.get("language_score", 0.0) < MIN_LANGUAGE_SCORE:
        return False
    for k in ("text", "content", "body", "document"):
        if isinstance(ex.get(k), str) and ex[k].strip():
            return True
    return False

def to_text(ex):
    for k in ("text","content","body","document"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return {"text": v}
    return {"text": ""}

class TokenBlockIterableDataset(IterableDataset):
    def __init__(self, stream, encoder, max_length=1024, add_eot=True):
        self.stream = stream
        self.max_length = max_length
        self.add_eot = add_eot
        self.enc = encoder
        self.EOT = encoder.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def __iter__(self):
        it = self.stream
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1: # multithread dataloader 
            it = self.stream.shard(num_shards=wi.num_workers, index=wi.id, contiguous=True)

        buf = []
        for ex in it: # generator
            if not ex:
                continue
            text = ex.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            toks = self.enc.encode(text)
            # append EOT token to the end of text
            if self.add_eot:
                toks.append(self.EOT)

            buf.extend(toks)
            while len(buf) >= self.max_length + 1:
                x = torch.tensor(buf[:self.max_length], dtype=torch.long)
                y = torch.tensor(buf[1:self.max_length+1], dtype=torch.long)
                yield x, y
                del buf[:self.max_length]    


class FixedPairsDataset(TorchDataset):
    def __init__(self, pairs):
        self.pairs = pairs  # list of (x, y) tensors

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def materialize_from_loader(loader, max_batches=32):
    pairs = []
    for i, (x, y) in enumerate(loader):
        pairs.extend(list(zip(x, y)))
        if (i + 1) >= max_batches:
            break
    return FixedPairsDataset(pairs)


def make_fixed_eval_loaders(train_loader,
                            val_loader,
                            max_train_batches=8,
                            max_val_batches=16):
    # 继承原 loader 的关键参数
    train_bs = getattr(train_loader, "batch_size", 1) or 1
    val_bs   = getattr(val_loader, "batch_size", 1) or 1
    pin_train = getattr(train_loader, "pin_memory", False)
    pin_val   = getattr(val_loader, "pin_memory", False)

    # 固化前 N 个 batch
    fixed_train_eval = materialize_from_loader(train_loader, max_batches=max_train_batches)
    fixed_val_eval   = materialize_from_loader(val_loader,   max_batches=max_val_batches)

    # 包成新的（map-style）DataLoader
    train_eval_loader = DataLoader(
        fixed_train_eval,
        batch_size=train_bs,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_train,
    )
    val_eval_loader = DataLoader(
        fixed_val_eval,
        batch_size=val_bs,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_val,
    )
    return train_eval_loader, val_eval_loader

def get_fineweb_loaders(max_length=1024, batch_size=32, val_mod=100, num_workers=0, add_eot=True):
    """
    train_loader, val_loader
    """
    fineweb = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    fineweb = fineweb.filter(is_en_high_conf)
    fineweb = fineweb.map(to_text)

    train_stream = fineweb.filter(lambda ex, idx: (idx % val_mod) != 0, with_indices=True)
    val_stream   = fineweb.filter(lambda ex, idx: (idx % val_mod) == 0, with_indices=True)

    enc = tiktoken.get_encoding("gpt2")
    train_ds = TokenBlockIterableDataset(train_stream, encoder=enc, max_length=max_length, add_eot=add_eot)
    val_ds   = TokenBlockIterableDataset(val_stream,   encoder=enc, max_length=max_length, add_eot=add_eot)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

if __name__ == "__main__":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_TOKEN"] = "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"

    # max_length = 1024
    # batch_size = 32
    # VAL_MOD = 100  # 1%
    # num_workers = 4

    # fineweb = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    # fineweb = fineweb.filter(is_en_high_conf)
    # fineweb = fineweb.map(to_text)

    # train_stream = fineweb.filter(lambda ex, idx: (idx % VAL_MOD) != 0, with_indices=True)
    # val_stream   = fineweb.filter(lambda ex, idx: (idx % VAL_MOD) == 0, with_indices=True)

    # train_ds = TokenBlockIterableDataset(train_stream, encoder=tiktoken.get_encoding("gpt2"), max_length=max_length, add_eot=True)
    # validation_ds = TokenBlockIterableDataset(val_stream, encoder=tiktoken.get_encoding("gpt2"), max_length=max_length, add_eot=True)

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=False,        
    #     drop_last=True, # if last batch < batch_size then drop it
    #     num_workers=num_workers, 
    #     pin_memory=True
    # )

    # val_loader = DataLoader(
    #     validation_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )



    # train_loader, val_loader = get_fineweb_loaders(num_workers=4)

 
    # start = time.time()
    # n_tokens = 0
    # n_batches = 0
    # for xb, yb in val_loader:
    #     n_batches += 1
    #     n_tokens += xb.numel()  # 每个 batch 的 token 数
    #     if n_batches % 100 == 0:
    #         elapsed = time.time() - start
    #         print(f"[{n_batches} batches] {n_tokens} tokens, {n_tokens/elapsed:.1f} tok/s")

    train_loader, val_loader = get_fineweb_loaders(
        max_length=1024, batch_size=32, num_workers=4, add_eot=True
    )

    train_eval_loader, val_eval_loader = make_fixed_eval_loaders(
        train_loader, val_loader, max_train_batches=8, max_val_batches=16
    )

