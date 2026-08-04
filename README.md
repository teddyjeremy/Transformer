# Transformer English to French

A PyTorch implementation of the Transformer architecture for English-to-French neural machine translation using the OPUS Books dataset.

This project is inspired by the architecture introduced in:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). *Attention Is All You Need*.

[Read the paper on arXiv](https://arxiv.org/abs/1706.03762)

## Project Structure

```text
Transformer/
├── config.py
├── tokenizer.py
├── dataset.py
├── model.py
├── train.py
├── translate.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   └── Transformer_English_to_French.ipynb
└── tests/
    └── test_model.py
```

## Core Components

`config.py` contains training, model, tokenizer, checkpoint, and TensorBoard configuration.

`tokenizer.py` handles WordLevel tokenizer creation, vocabulary training, tokenizer persistence, and tokenizer loading.

`dataset.py` prepares bilingual examples, creates encoder and decoder inputs, builds padding masks, and provides the causal attention mask.

`model.py` implements the Transformer encoder-decoder architecture with embeddings, sinusoidal positional encoding, multi-head attention, feed-forward networks, residual connections, normalization, and vocabulary projection.

`train.py` handles OPUS Books loading, dataset splitting, training, greedy decoding, validation metrics, TensorBoard logging, and checkpoint management.

`translate.py` provides reusable checkpoint loading and English-to-French inference functions without coupling inference to the training loop.

`notebooks/Transformer_English_to_French.ipynb` provides an interactive walkthrough of the model, masks, and tensor shapes.

`tests/test_model.py` contains shape and causal-mask tests for the core architecture.

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

The first run builds WordLevel tokenizers from the OPUS Books training data. Subsequent runs reuse the generated tokenizer files.

The training pipeline creates a deterministic 90/10 training-validation split, trains the Transformer with teacher forcing, evaluates generated translations, records training and validation metrics, and saves model checkpoints.

## Translation

After training, inference can be performed from Python:

```python
from config import get_config
from translate import load_translation_model, translate_text

config = get_config()
model, tokenizer_src, tokenizer_tgt, device = load_translation_model(config)

translation = translate_text(
    "Hello, how are you?",
    model,
    tokenizer_src,
    tokenizer_tgt,
    config,
    device
)
```

## TensorBoard

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

## Testing

Run the model tests with:

```bash
pytest
```

## Checkpoints

Model checkpoints are stored under the configured model directory and are excluded from version control. Set `preload` in `config.py` to resume from a particular checkpoint or use `latest` to load the most recent checkpoint.

## Architecture

The implementation follows the encoder-decoder Transformer design with:

- Token embeddings scaled by the square root of the model dimension
- Sinusoidal positional encoding
- Multi-head self-attention
- Decoder masked self-attention
- Encoder-decoder cross-attention
- Position-wise feed-forward networks
- Pre-normalized residual connections
- Stacked encoder and decoder blocks
- Linear vocabulary projection
- Xavier uniform parameter initialization

## Data

The project uses the OPUS Books dataset through the Hugging Face `datasets` library. The default configuration trains an English-to-French translation model.

## Attribution

The Transformer architecture implemented here is based on the work presented in *Attention Is All You Need* by Vaswani et al. The project also uses the implementation approach demonstrated by [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) as a reference while adapting the project structure and implementation for this repository.

Paper: https://arxiv.org/abs/1706.03762

Reference implementation: https://github.com/hkproj/pytorch-transformer
