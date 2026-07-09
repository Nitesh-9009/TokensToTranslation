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
TokensToTranslation/
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
cd TokensToTranslation
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

## Sample Results

With only 5k pairs and 10 epochs the model is small, but it clearly learns the
mapping for common short sentences:

| English | German (predicted) |
|---------|--------------------|
| i love india . | ich liebe indien . |
| she is reading a book . | sie liest ein buch . |
| we are going home . | wir gehen nach hause . |
| the weather is nice today . | das wetter ist heute schön . |
| he drinks coffee every morning . | er trinkt jeden morgen kaffee . |

(Exact output varies per run — training is stochastic and the corpus is small.)

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
