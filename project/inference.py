"""
inference.py
------------
Load a trained translator.pt and translate English sentences into German.

    python inference.py            # runs a handful of demo sentences
    python inference.py "I love India."

Greedy decoding, no teacher forcing -- the decoder is fed its own previous
prediction until it emits <eos> (or we hit max_len).
"""

import sys

import torch

import config
from utils import (
    Vocabulary,
    normalize,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)
from model import Encoder, Decoder, Seq2Seq


def _rebuild_vocab(word2idx, idx2word):
    """Recreate a Vocabulary object from the saved dicts (skip .build())."""
    vocab = Vocabulary()
    vocab.word2idx = word2idx
    # torch.save turns int keys into ... ints, but json-ish round trips can make
    # them strings, so coerce to be safe.
    vocab.idx2word = {int(k): v for k, v in idx2word.items()}
    return vocab


def load_model(path=config.MODEL_PATH, device=config.DEVICE):
    """Restore the model + both vocabularies from a checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]

    src_vocab = _rebuild_vocab(ckpt["src_word2idx"], ckpt["src_idx2word"])
    trg_vocab = _rebuild_vocab(ckpt["trg_word2idx"], ckpt["trg_idx2word"])

    encoder = Encoder(
        len(src_vocab), cfg["emb_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"],
    )
    decoder = Decoder(
        len(trg_vocab), cfg["emb_dim"], cfg["hidden_dim"],
        cfg["num_layers"], cfg["dropout"],
    )
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, src_vocab, trg_vocab, cfg["max_len"]


@torch.no_grad()
def translate_sentence(sentence, model, src_vocab, trg_vocab, max_len=20,
                       device=config.DEVICE):
    """Translate a single English sentence into German (greedy decoding)."""
    model.eval()

    # encode source
    sos = src_vocab.word2idx[SOS_TOKEN]
    eos = src_vocab.word2idx[EOS_TOKEN]
    tokens = src_vocab.numericalize(normalize(sentence))
    src_ids = [sos] + tokens + [eos]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    hidden, cell = model.encoder(src_tensor)

    # start decoding from the target <sos>
    trg_sos = trg_vocab.word2idx[SOS_TOKEN]
    trg_eos = trg_vocab.word2idx[EOS_TOKEN]
    input_token = torch.tensor([trg_sos], dtype=torch.long, device=device)

    result = []
    for _ in range(max_len):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        pred = output.argmax(1)
        token_id = pred.item()
        if token_id == trg_eos:
            break
        result.append(token_id)
        input_token = pred

    words = trg_vocab.decode(result)
    return " ".join(words)


DEMO_SENTENCES = [
    "I love India.",
    "She is reading a book.",
    "We are going home.",
    "The weather is nice today.",
    "He drinks coffee every morning.",
]


def main():
    model, src_vocab, trg_vocab, max_len = load_model()

    sentences = sys.argv[1:] if len(sys.argv) > 1 else DEMO_SENTENCES
    for sentence in sentences:
        german = translate_sentence(sentence, model, src_vocab, trg_vocab, max_len)
        print(f"EN: {sentence}")
        print(f"DE: {german}\n")


if __name__ == "__main__":
    main()
