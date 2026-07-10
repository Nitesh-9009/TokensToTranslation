

import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Embeds the source tokens and runs them through an LSTM.

    We throw away the per-timestep outputs and only keep the final hidden /
    cell state -- that compressed context is what the decoder gets.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))       # [batch, src_len, emb]
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """Generates the target sentence one token at a time.

    A single forward() call handles exactly one time step; the Seq2Seq module
    is responsible for looping over the sequence.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch] -> add a time dim so the LSTM is happy
        input_token = input_token.unsqueeze(1)              # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))         # [batch, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Encoder + Decoder with configurable teacher forcing."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len]   trg: [batch, trg_len]
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        hidden, cell = self.encoder(src)

        # first decoder input is the <sos> token of every target sequence
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            # decide whether to feed the ground-truth token or our own guess
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs
