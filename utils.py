"""
utils.py
--------
Small helper module that takes care of everything *before* the model sees the
data: downloading the corpus, cleaning the text, tokenising it and building the
two vocabularies (one for English, one for German).

I kept the tokenizer deliberately dumb (plain whitespace) because the whole
point of the SoC project was to build the seq2seq stack ourselves rather than
lean on spaCy / HF tokenizers.
"""

import os
import re
import zipfile
import urllib.request

# ----------------------------------------------------------------------------
# Special tokens. Order matters -> these get indices 0..3 so PAD stays 0.
# ----------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

DATA_URL = "https://www.manythings.org/anki/deu-eng.zip"


def download_dataset(data_dir="data"):
    """Download + unzip the ManyThings deu-eng corpus if it isn't there yet.

    Returns the path to the extracted ``deu.txt`` file.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "deu-eng.zip")
    txt_path = os.path.join(data_dir, "deu.txt")

    if os.path.exists(txt_path):
        print(f"Dataset already present at {txt_path}")
        return txt_path

    print("Downloading dataset ...")
    # manythings.org blocks the default urllib user-agent, so spoof a browser.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(DATA_URL, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return txt_path


def normalize(text):
    """Lower-case, trim, and put spaces around punctuation.

    Doing this before whitespace-splitting means "hello!" becomes two tokens
    ("hello", "!") instead of one weird token.
    """
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:])", r" \1 ", text)      # isolate punctuation
    text = re.sub(r"[^a-zäöüß.!?,;:']+", " ", text)  # keep letters + umlauts
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Whitespace tokenizer. Assumes text was already normalised."""
    return text.split()


def load_pairs(txt_path, num_pairs=5000):
    """Read the tab separated file and return a list of (eng, deu) tuples.

    The ManyThings file is ``english<TAB>german<TAB>attribution`` per line, so
    we only care about the first two columns.
    """
    pairs = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            eng = normalize(parts[0])
            deu = normalize(parts[1])
            if eng and deu:
                pairs.append((eng, deu))
            if len(pairs) >= num_pairs:
                break
    return pairs


class Vocabulary:
    """Maps tokens <-> integer ids for a single language.

    Nothing fancy: a frequency filter (``min_freq``) and the four special
    tokens wired in from the start.
    """

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self._freq = {}

        # reserve the special tokens first
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

    def _add_token(self, token):
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build(self, sentences):
        """Count words across all sentences, then register the frequent ones."""
        for sentence in sentences:
            for token in tokenize(sentence):
                self._freq[token] = self._freq.get(token, 0) + 1

        for token, freq in self._freq.items():
            if freq >= self.min_freq:
                self._add_token(token)
        return self

    def numericalize(self, sentence):
        """Turn a raw sentence into a list of ids (unknown words -> <unk>)."""
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(tok, unk) for tok in tokenize(sentence)]

    def decode(self, ids):
        """Ids -> tokens, dropping the special markers so output reads cleanly."""
        skip = {self.word2idx[t] for t in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)}
        return [self.idx2word[i] for i in ids if i not in skip]

    def __len__(self):
        return len(self.word2idx)
