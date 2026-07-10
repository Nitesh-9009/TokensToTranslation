"""
train.py
--------
End-to-end training script. Run it directly:

    python train.py

It downloads the data, builds the vocabs, trains the seq2seq model for a few
epochs with teacher forcing + gradient clipping, and dumps everything needed
for inference into translator.pt.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils import (
    download_dataset,
    load_pairs,
    Vocabulary,
    PAD_TOKEN,
)
from dataset import TranslationDataset, make_collate_fn
from model import Encoder, Decoder, Seq2Seq


def train_one_epoch(model, loader, optimizer, criterion, clip, tf_ratio):
    """Single pass over the data. Returns the average loss."""
    model.train()
    epoch_loss = 0.0

    for src, trg in loader:
        src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)

        # skip the <sos> column, then flatten for CrossEntropyLoss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_flat)
        loss.backward()

        # clip to keep the LSTM gradients from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx):
    """Validation pass (no teacher forcing). Returns (loss, token_accuracy).

    Token accuracy = fraction of non-padding target tokens the model predicts
    correctly. It's a simple, cheap proxy for how good the translations are.
    """
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for src, trg in loader:
        src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

        output = model(src, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        flat_out = output[:, 1:].reshape(-1, output_dim)
        flat_trg = trg[:, 1:].reshape(-1)

        epoch_loss += criterion(flat_out, flat_trg).item()

        # accuracy over the real (non-pad) tokens only
        preds = flat_out.argmax(1)
        mask = flat_trg != pad_idx
        correct += (preds[mask] == flat_trg[mask]).sum().item()
        total += mask.sum().item()

    return epoch_loss / len(loader), correct / max(total, 1)


def build_everything():
    """Wire up data, vocabs and model. Returns the objects train() needs."""
    txt_path = download_dataset("data")
    pairs = load_pairs(txt_path, num_pairs=config.NUM_PAIRS)
    print(f"Loaded {len(pairs)} sentence pairs")

    # hold out 10% as a dev set so we can measure accuracy on unseen sentences
    split = int(0.9 * len(pairs))
    train_pairs, dev_pairs = pairs[:split], pairs[split:]
    print(f"Train: {len(train_pairs)} | Dev: {len(dev_pairs)}")

    # vocab is built only on the training pairs (dev must stay unseen)
    src_vocab = Vocabulary(min_freq=config.MIN_FREQ).build([p[0] for p in train_pairs])
    trg_vocab = Vocabulary(min_freq=config.MIN_FREQ).build([p[1] for p in train_pairs])
    print(f"English vocab: {len(src_vocab)} | German vocab: {len(trg_vocab)}")

    pad_idx = src_vocab.word2idx[PAD_TOKEN]
    collate = make_collate_fn(pad_idx)

    train_ds = TranslationDataset(train_pairs, src_vocab, trg_vocab, max_len=config.MAX_LEN)
    dev_ds = TranslationDataset(dev_pairs, src_vocab, trg_vocab, max_len=config.MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate)

    encoder = Encoder(
        len(src_vocab), config.EMB_DIM, config.HIDDEN_DIM,
        config.NUM_LAYERS, config.DROPOUT,
    )
    decoder = Decoder(
        len(trg_vocab), config.EMB_DIM, config.HIDDEN_DIM,
        config.NUM_LAYERS, config.DROPOUT,
    )
    model = Seq2Seq(encoder, decoder, config.DEVICE).to(config.DEVICE)

    return model, train_loader, dev_loader, src_vocab, trg_vocab, pad_idx


def main():
    model, train_loader, dev_loader, src_vocab, trg_vocab, pad_idx = build_everything()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Training on {config.DEVICE} for {config.EPOCHS} epochs\n")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            config.CLIP, config.TEACHER_FORCING_RATIO,
        )
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, pad_idx)
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
              f"train loss {train_loss:.4f} | dev loss {dev_loss:.4f} | "
              f"dev token-acc {dev_acc * 100:.2f}%")

    # bundle weights + vocabs so inference.py is fully self-contained
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_word2idx": src_vocab.word2idx,
            "src_idx2word": src_vocab.idx2word,
            "trg_word2idx": trg_vocab.word2idx,
            "trg_idx2word": trg_vocab.idx2word,
            "config": {
                "emb_dim": config.EMB_DIM,
                "hidden_dim": config.HIDDEN_DIM,
                "num_layers": config.NUM_LAYERS,
                "dropout": config.DROPOUT,
                "max_len": config.MAX_LEN,
            },
        },
        config.MODEL_PATH,
    )
    print(f"\nSaved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
