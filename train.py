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
def evaluate(model, loader, criterion):
    """Validation-style pass with teacher forcing switched off."""
    model.eval()
    epoch_loss = 0.0

    for src, trg in loader:
        src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

        output = model(src, trg, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_flat)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def build_everything():
    """Wire up data, vocabs and model. Returns the objects train() needs."""
    txt_path = download_dataset("data")
    pairs = load_pairs(txt_path, num_pairs=config.NUM_PAIRS)
    print(f"Loaded {len(pairs)} sentence pairs")

    eng_sentences = [p[0] for p in pairs]
    deu_sentences = [p[1] for p in pairs]

    src_vocab = Vocabulary(min_freq=config.MIN_FREQ).build(eng_sentences)
    trg_vocab = Vocabulary(min_freq=config.MIN_FREQ).build(deu_sentences)
    print(f"English vocab: {len(src_vocab)} | German vocab: {len(trg_vocab)}")

    dataset = TranslationDataset(pairs, src_vocab, trg_vocab, max_len=config.MAX_LEN)
    pad_idx = src_vocab.word2idx[PAD_TOKEN]
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(pad_idx),
    )

    encoder = Encoder(
        len(src_vocab), config.EMB_DIM, config.HIDDEN_DIM,
        config.NUM_LAYERS, config.DROPOUT,
    )
    decoder = Decoder(
        len(trg_vocab), config.EMB_DIM, config.HIDDEN_DIM,
        config.NUM_LAYERS, config.DROPOUT,
    )
    model = Seq2Seq(encoder, decoder, config.DEVICE).to(config.DEVICE)

    return model, loader, src_vocab, trg_vocab, pad_idx


def main():
    model, loader, src_vocab, trg_vocab, pad_idx = build_everything()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Training on {config.DEVICE} for {config.EPOCHS} epochs\n")
    for epoch in range(1, config.EPOCHS + 1):
        loss = train_one_epoch(
            model, loader, optimizer, criterion,
            config.CLIP, config.TEACHER_FORCING_RATIO,
        )
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | loss {loss:.4f}")

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
