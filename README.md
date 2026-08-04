# Transformer

A PyTorch implementation of the Transformer architecture for English-to-French translation using the OPUS Books dataset.

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
