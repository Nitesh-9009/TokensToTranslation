"""
dataset.py
----------
PyTorch ``Dataset`` + a ``collate_fn`` that pads a batch to the longest
sequence in that batch. Every source/target sequence is wrapped with
<sos> ... <eos> so the decoder knows where to start and stop.
"""

import torch
from torch.utils.data import Dataset

from utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class TranslationDataset(Dataset):
    """Holds numericalised (source, target) id tensors.

    max_len caps sequence length so a couple of very long outliers don't blow
    up the padding for the whole batch.
    """

    def __init__(self, pairs, src_vocab, trg_vocab, max_len=20):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

        self.sos = trg_vocab.word2idx[SOS_TOKEN]
        self.eos = trg_vocab.word2idx[EOS_TOKEN]

        # keep the id lists in memory; the corpus is tiny so this is fine
        self.data = []
        for eng, deu in pairs:
            src_ids = self._encode(eng, src_vocab)
            trg_ids = self._encode(deu, trg_vocab)
            self.data.append((src_ids, trg_ids))

    def _encode(self, sentence, vocab):
        ids = vocab.numericalize(sentence)[: self.max_len - 2]
        return [self.sos] + ids + [self.eos]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            trg_ids, dtype=torch.long
        )


def make_collate_fn(pad_idx):
    """Return a collate_fn closure that pads to the batch max length."""

    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)

        src_max = max(len(s) for s in src_batch)
        trg_max = max(len(t) for t in trg_batch)

        def pad(seq, length):
            return torch.cat(
                [seq, torch.full((length - len(seq),), pad_idx, dtype=torch.long)]
            )

        src_padded = torch.stack([pad(s, src_max) for s in src_batch])
        trg_padded = torch.stack([pad(t, trg_max) for t in trg_batch])
        return src_padded, trg_padded

    return collate_fn
