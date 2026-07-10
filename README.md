# TokensToTranslation — Deep Learning SoC by Nitesh Patel

This repo has my Deep Learning Summer of Code work. It goes week by week and
finally ends with the main project (English → German translator).

The main branch has **4 things**:

| # | File / Folder | What is inside |
|---|---------------|----------------|
| 1 | [firstWeek_SOC.ipynb](firstWeek_SOC.ipynb) | Micrograd — apna chota autograd engine, backprop, tanh, Neuron / Layer / MLP |
| 2 | [secondWeek_SOC.ipynb](secondWeek_SOC.ipynb) | makemore part 2 & 3 — MLP (Bengio 2003), character embeddings, weight init, batch norm + assignment exercises |
| 3 | [thirdWeek_SOC.ipynb](thirdWeek_SOC.ipynb) | makemore part 4 & 5 — becoming a backprop ninja (manual gradients) + WaveNet |
| 4 | [project/](project/) | Final project — English → German seq2seq LSTM translator (whole code + notebook + README) |

---

## How to run

- **Week notebooks** — open any `*_SOC.ipynb` in Google Colab (Colab badge on
  top of each) and run the cells top to bottom.
- **Project** — everything for the translator is inside [project/](project/).
  See [project/README.md](project/README.md) for details, or open
  [project/TokensToTranslation.ipynb](project/TokensToTranslation.ipynb) in Colab.

Each week builds towards the last one — micrograd gives the backprop idea,
makemore gives embeddings + training tricks, RNN/LSTM/attention give the
sequence models, and the project puts encoder + decoder together for translation.
