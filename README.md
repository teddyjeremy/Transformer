# Transformer

A PyTorch implementation of the Transformer architecture for English-to-French translation using the OPUS Books dataset.

This project is based on the Transformer architecture introduced in *Attention Is All You Need* by Ashish Vaswani et al.:

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Project Structure

```text
Transformer/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

The training pipeline builds WordLevel tokenizers, loads OPUS Books, creates train and validation datasets, trains the Transformer, evaluates translations, logs metrics to TensorBoard, and saves checkpoints.

## TensorBoard

```bash
tensorboard --logdir runs
```

## Architecture

The model contains input embeddings, sinusoidal positional encoding, multi-head self-attention, encoder-decoder cross-attention, feed-forward networks, pre-normalized residual connections, stacked encoder and decoder blocks, and a projection layer.

## Configuration

Training and model settings are defined in `config.py`. Checkpoints are stored separately from source code and are excluded from Git through `.gitignore`.

## Reference

The original Transformer architecture is described in:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). *Attention Is All You Need*.

https://arxiv.org/abs/1706.03762
