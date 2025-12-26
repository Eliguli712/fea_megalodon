#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Char-level LSTM for your 镜塔 / 魔女 / 宗裁 语系文本
--------------------------------------------------
Usage (train):
    python train_char_lstm.py --corpus corpus_me.txt --epochs 50

Usage (generate from trained model):
    python train_char_lstm.py --generate \
        --checkpoint mirrorwitch_char_lstm.pt \
        --start "鏡塔残響\n" \
        --length 400 \
        --temperature 0.8
"""

import argparse
import os
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
#  Dataset
# ----------------------------

class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        """
        text: a string
        seq_len: length of input sequence
        """
        super().__init__()
        self.text = text
        self.seq_len = seq_len

        # Build vocab
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Encode text as list of ints
        self.ids = [self.stoi[ch] for ch in text]

    def __len__(self):
        # one step-ahead prediction
        return len(self.ids) - self.seq_len - 1

    def __getitem__(self, idx):
        """
        Returns:
            x: (seq_len,) long
            y: (seq_len,) long, x shifted by +1
        """
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + 1 + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ----------------------------
#  Model
# ----------------------------

class CharLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size=256,
                 hidden_size=512,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (B, T)
        hidden: (h0, c0) or None
        returns:
            logits: (B, T, vocab_size)
            hidden: (hn, cn)
        """
        emb = self.emb(x)              # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.fc(out)          # (B, T, V)
        return logits, hidden


# ----------------------------
#  Utility: sampling
# ----------------------------

def sample_from_logits(logits, temperature=1.0):
    """
    logits: (vocab_size,) tensor
    temperature: float > 0
    """
    if temperature <= 0:
        # greedy
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def generate_text(model,
                  stoi,
                  itos,
                  device,
                  start_text="鏡塔残響\n",
                  length=400,
                  temperature=0.8):
    """
    Generate text from a trained model.
    """
    model.eval()
    # Encode start_text
    seq = [stoi[ch] for ch in start_text if ch in stoi]
    if not seq:
        # fallback: random start
        seq = [random.choice(list(stoi.values()))]

    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    hidden = None
    out_chars = [itos[idx] for idx in seq]

    with torch.no_grad():
        # Prime the model with the start_text
        logits, hidden = model(input_ids, hidden)

        # Now generate new chars
        last_id = input_ids[0, -1].unsqueeze(0).unsqueeze(0)  # (1, 1)
        for _ in range(length):
            logits, hidden = model(last_id, hidden)  # logits: (1, 1, V)
            next_logits = logits[0, -1, :]           # (V,)
            next_id = sample_from_logits(next_logits, temperature=temperature)
            out_chars.append(itos[next_id])
            last_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(out_chars)


# ----------------------------
#  Training loop
# ----------------------------

def train(args):
    # Load corpus
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    # Basic cleaning（按需调整）
    text = text.replace("\r", "")
    # 你如果想保留所有换行，就不要删 "\n"
    if args.strip_empty_lines:
        # 去一些连续空行
        lines = [ln for ln in text.split("\n") if ln.strip() != ""]
        text = "\n".join(lines)

    dataset = CharDataset(text, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = CharLSTM(
        vocab_size=len(dataset.stoi),
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Vocab size: {len(dataset.stoi)}")
    print(f"Total chars in corpus: {len(dataset.ids)}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)  # (B, T)
            y = y.to(device)  # (B, T)

            optimizer.zero_grad()
            logits, _ = model(x)  # (B, T, V)
            # flatten
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch} Step {global_step} | loss={avg:.4f}")

        avg_loss = total_loss / (batch_idx + 1)
        print(f"\n[Epoch {epoch}] avg loss = {avg_loss:.4f}\n")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model,
                dataset.stoi,
                dataset.itos,
                args,
                path=args.checkpoint,
            )
            print(f"Checkpoint saved to {args.checkpoint} (best_loss={best_loss:.4f})")

        # 每个 epoch 后随手生成一点玩
        if args.sample_each_epoch:
            sample = generate_text(
                model,
                dataset.stoi,
                dataset.itos,
                device,
                start_text=args.sample_start,
                length=args.sample_len,
                temperature=args.sample_temp,
            )
            print("=== Sample ===")
            print(sample)
            print("==============\n")


def save_checkpoint(model, stoi, itos, args, path="mirrorwitch_char_lstm.pt"):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "emb_size": args.emb_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "seq_len": args.seq_len,
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    cfg = ckpt["config"]

    model = CharLSTM(
        vocab_size=len(stoi),
        emb_size=cfg["emb_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stoi, itos


# ----------------------------
#  Argparse / main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="corpus_me.txt",
                        help="Path to training corpus (UTF-8 text).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="mirrorwitch_char_lstm.pt")
    parser.add_argument("--strip_empty_lines", action="store_true",
                        help="If set, remove completely empty lines from corpus.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")

    # sampling options during training
    parser.add_argument("--sample_each_epoch", action="store_true",
                        help="Generate a sample text after each epoch.")
    parser.add_argument("--sample_start", type=str, default="鏡塔残響\n")
    parser.add_argument("--sample_len", type=int, default=300)
    parser.add_argument("--sample_temp", type=float, default=0.8)

    # generation-only mode
    parser.add_argument("--generate", action="store_true",
                        help="Generation mode: requires --checkpoint.")
    parser.add_argument("--start", type=str, default="鏡塔残響\n",
                        help="Start text for generation.")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--gen_checkpoint", type=str, default=None,
                        help="Checkpoint to load for generation. "
                             "If not set, uses --checkpoint.")

    args = parser.parse_args()

    if args.generate:
        ckpt_path = args.gen_checkpoint or args.checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"Using device: {device}")
        print(f"Loading checkpoint: {ckpt_path}")
        model, stoi, itos = load_checkpoint(ckpt_path, device)
        text = generate_text(
            model,
            stoi,
            itos,
            device,
            start_text=args.start,
            length=args.length,
            temperature=args.temperature,
        )
        print(text)
    else:
        train(args)


if __name__ == "__main__":
    main()