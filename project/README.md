# TokensToTranslation — by Nitesh Patel

An English → German neural machine translator built with a plain PyTorch
sequence-to-sequence LSTM. No HuggingFace pipelines, no pretrained translation
models — the encoder, decoder and the seq2seq loop (with teacher forcing) are
all written by hand.

This was my final project for the **Deep Learning Summer of Code**, after
working through micrograd, character embeddings, MLPs, the makemore series,
batch-norm / init, RNNs, LSTMs, attention and transformers. The idea was to
take everything from those and actually build a translator end to end.

---

## Project Description

Given an English sentence, the model produces its German translation one token
at a time. Under the hood it's the classic Sutskever et al. seq2seq setup:

* an **Encoder** LSTM reads the whole English sentence and compresses it into a
  final hidden/cell state,
* a **Decoder** LSTM starts from that state and generates German tokens
  autoregressively until it emits `<eos>`.

During training the decoder is sometimes fed the ground-truth previous token
(*teacher forcing*) and sometimes its own prediction, which stabilises early
learning. At inference time teacher forcing is off and decoding is greedy.

---

## Architecture

```
English sentence
      │
      ▼
  Embedding (128)
      │
      ▼
  Encoder LSTM (hidden 256)  ──►  (hidden, cell)
                                       │
                                       ▼
                            Decoder LSTM (hidden 256)
                                       │
                                 Linear → vocab
                                       │
                                       ▼
                               German tokens
```

| Component | Layers |
|-----------|--------|
| Encoder   | `nn.Embedding` → `nn.LSTM` |
| Decoder   | `nn.Embedding` → `nn.LSTM` → `nn.Linear` |
| Seq2Seq   | Encoder + Decoder + teacher forcing loop |

Default hyper-parameters:

| Param | Value |
|-------|-------|
| Embedding dim | 128 |
| Hidden dim | 256 |
| LSTM layers | 1 |
| Dropout | 0.2 |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Epochs | 10 |
| Loss | `CrossEntropyLoss(ignore_index=<pad>)` |
| Gradient clip | 1.0 |

---

## Dataset

[ManyThings.org Anki deu-eng](https://www.manythings.org/anki/deu-eng.zip)
tab-separated English/German sentence pairs. The notebook downloads and
extracts the zip automatically and uses the first ~5000 pairs so a full run
finishes in a few minutes.

Preprocessing:

* lower-case + trim
* spaces inserted around punctuation
* plain whitespace tokenization (no spaCy / SentencePiece)
* manual vocabularies with `<pad> <sos> <eos> <unk>` special tokens

---

## Folder Structure

```
TokensToTranslation/                  # repo root (also holds the weekly SoC notebooks)
└── project/
    ├── data/                     # corpus is downloaded here (gitignored)
    ├── config.py                 # all hyper-parameters in one place
    ├── utils.py                  # download, cleaning, tokenizer, Vocabulary
    ├── dataset.py                # TranslationDataset + padding collate_fn
    ├── model.py                  # Encoder, Decoder, Seq2Seq
    ├── train.py                  # training loop, saves translator.pt
    ├── inference.py              # load model + translate_sentence()
    ├── requirements.txt
    ├── README.md
    └── TokensToTranslation.ipynb # full Colab notebook (everything inline)
```

---

## Installation

```bash
git clone https://github.com/Nitesh-9009/TokensToTranslation.git
cd TokensToTranslation/project
pip install -r requirements.txt
```

Or just open `TokensToTranslation.ipynb` in Google Colab and run the cells top
to bottom — it installs nothing extra beyond what Colab already ships.

---

## Training

```bash
python train.py
```

This downloads the data, builds the vocabularies, trains for 10 epochs and
writes `translator.pt` (weights **and** both vocabularies, so inference needs
nothing else).

---

## Inference

```bash
# a few built-in demo sentences
python inference.py

# or your own
python inference.py "I love India."
```

Programmatic use:

```python
from inference import load_model, translate_sentence

model, src_vocab, trg_vocab, max_len = load_model("translator.pt")
print(translate_sentence("I love India.", model, src_vocab, trg_vocab, max_len))
```

---

## Results & Accuracy

I keep a 10% held-out dev set (500 sentence pairs the model never trains on) and
measure **token-level accuracy** on it — the fraction of non-padding target words
the model predicts correctly. I also watch the loss.

Training run (10 epochs, 4500 train / 500 dev pairs, CPU):

| Epoch | Train loss | Dev loss | Dev token-acc |
|-------|-----------|----------|---------------|
| 1  | 4.62 | 3.83 | 41.9% |
| 4  | 3.03 | 3.21 | 52.2% |
| 7  | 2.41 | 3.12 | 56.1% |
| 10 | 2.03 | 3.06 | **58.2%** |

So the accuracy goes up every epoch — the model is actually learning, not just
memorising. Numbers vary a little each run since training is random.

Some real outputs from the trained model (`python inference.py`):

| English | German (predicted) |
|---------|--------------------|
| I am happy. | ich bin es . |
| He is here. | er ist ein . |
| Tom is my friend. | tom ist eine . . |
| We are ready. | wir haben ihn . |
| I love India. | ich liebe . . |

It gets the sentence structure and pronouns right (ich / er / wir / tom …) but
the content words and longer sentences are still shaky. That makes sense — it's a
small model, only 10 epochs, and there's no attention yet (the whole English
sentence is squeezed into one vector). Adding attention is the main thing that
would push the accuracy up.

### How to check the accuracy yourself

Just run the training — it prints the dev loss and dev token-accuracy after every
epoch:

```bash
python train.py
```

```
Epoch 10/10 | train loss 2.0321 | dev loss 3.0628 | dev token-acc 58.19%
```

The token-accuracy logic lives in the `evaluate()` function in
[train.py](train.py): it runs the model on the dev set with teacher forcing off,
takes `argmax` over the logits, and compares against the target, skipping `<pad>`.
To eyeball quality instead of a number, translate your own sentences:

```bash
python inference.py "He is here." "I am tired."
```

---

## Future Improvements

* Add **attention** (Bahdanau/Luong) so the decoder isn't bottlenecked by a
  single context vector — this is the obvious next step given the SoC syllabus.
* Swap greedy decoding for **beam search**.
* Train on the **full** corpus (200k+ pairs) with more epochs.
* Subword tokenization (BPE) to shrink the vocab and handle rare words.
* Track **BLEU** on a held-out validation split instead of eyeballing outputs.
* Try a small **Transformer** encoder/decoder and compare.

---

## Notes

No pretrained translation weights are used anywhere. Only generic building
blocks (`nn.Embedding`, `nn.LSTM`, `nn.Linear`, `CrossEntropyLoss`, `Adam`) —
the translation logic itself is implemented in this repo.
