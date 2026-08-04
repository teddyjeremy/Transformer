# Transformer English to French

A from-scratch PyTorch implementation of the original encoder-decoder Transformer for English-to-French neural machine translation using OPUS Books.

The project implements the architecture introduced by Vaswani et al. in *Attention Is All You Need* and develops the complete training, evaluation, inference, visualization, testing, and notebook workflow around that architecture.

Paper: https://arxiv.org/abs/1706.03762

## Project Structure

```text
Transformer/
├── config.py
├── tokenizer.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── translate.py
├── visualization.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── Transformer_English_to_French.ipynb
│   ├── Training_Analysis.ipynb
│   ├── Validation_Analysis.ipynb
│   └── Inference_and_Attention.ipynb
├── tests/
│   └── test_model.py
└── .github/
    └── workflows/
        └── tests.yml
```

## Implementation

The model is implemented without `torch.nn.Transformer` so the principal Transformer components are explicit in the source code.

### Input pipeline

The OPUS Books English-French corpus is loaded through Hugging Face Datasets. Separate WordLevel tokenizers are trained for the source and target languages and persisted as JSON files.

Each training example produces:

- Source sequence: `[SOS] source tokens [EOS] [PAD] ...`
- Decoder input: `[SOS] target tokens [PAD] ...`
- Label: `target tokens [EOS] [PAD] ...`

Source and target vocabularies have independent special-token IDs. Padding is validated against the configured sequence length before tensors are created.

### Attention masks

The encoder receives a source padding mask. The decoder combines target padding with a lower-triangular causal mask. The causal mask prevents a target position from attending to future positions during teacher forcing.

### Embeddings

Token embeddings are scaled by `sqrt(d_model)`. Sinusoidal positional encodings are registered as buffers and added before the encoder or decoder stack.

### Multi-head attention

The implementation contains independent query, key, and value projections, head splitting, scaled dot-product attention, masking, dropout, head concatenation, and output projection.

For each head:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

Attention probabilities are retained on attention blocks for visualization and analysis.

### Encoder

Each encoder block contains multi-head self-attention, a residual connection with layer normalization and dropout, a position-wise feed-forward network, and a second residual connection.

### Decoder

Each decoder block contains masked multi-head self-attention, encoder-decoder cross-attention, and a position-wise feed-forward network. Each sublayer uses residual connections, layer normalization, and dropout.

### Feed-forward network

The position-wise network expands the representation from `d_model` to `d_ff`, applies ReLU and dropout, then projects back to `d_model`.

### Output projection

The decoder representation is projected to the target vocabulary size. Cross-entropy loss with label smoothing is calculated against the shifted target sequence while ignoring target padding tokens.

## Training

Create and activate an environment:

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

Train:

```bash
python train.py
```

The training pipeline handles tokenizers, dataset splitting, DataLoaders, model construction, AdamW optimization, gradient clipping, label smoothing, TensorBoard logging, validation metrics, and checkpoint persistence.

Default model settings are six encoder layers, six decoder layers, eight attention heads, `d_model=512`, `d_ff=2048`, dropout `0.1`, and sequence length `128`.

## Checkpointing

Checkpoints are written under the configured model directory and excluded from version control.

To resume from the latest checkpoint, set:

```python
"preload": "latest"
```

To resume from a specific checkpoint:

```python
"preload": "05"
```

The checkpoint contains model parameters, optimizer state, epoch, global step, and the training configuration used for the run.

## Evaluation

Run the evaluation pipeline after training:

```bash
python evaluate.py
```

The evaluation pipeline supports character error rate, word error rate, BLEU, greedy autoregressive decoding, and validation-set evaluation.

## Translation

`translate.py` provides reusable checkpoint loading, tokenization, source encoding, greedy autoregressive decoding, and English-to-French translation.

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

## Visualization

`visualization.py` provides reusable plots for:

- Training and validation loss
- CER, WER, and BLEU
- Source and target token-length distributions
- Multi-head attention matrices
- Autoregressive token confidence analysis

Figures can be displayed directly in notebooks or saved to disk for reports and experiments.

## Notebooks

The project contains separate notebooks for implementation, training analysis, validation, and inference.

### Transformer_English_to_French.ipynb

The main implementation notebook develops the Transformer from its mathematical components through the complete encoder-decoder model and translation pipeline.

### Training_Analysis.ipynb

The training notebook examines corpus size, vocabulary sizes, token-length distributions, model parameter counts, training history, and validation metrics.

### Validation_Analysis.ipynb

The validation notebook evaluates generated translations using CER, WER, and BLEU and provides attention visualization for encoder-decoder relationships.

### Inference_and_Attention.ipynb

The inference notebook runs multiple English-to-French examples, examines autoregressive decoding, tracks top-token probabilities at each decoding step, and inspects generated token sequences.

The notebooks are intended to provide reproducible experiments and visual evidence of how the Transformer operates rather than serving only as demonstrations of library APIs.

## TensorBoard

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

Training loss and validation metrics are recorded during training.

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover causal-mask correctness, attention masking, encoder and decoder tensor shapes, vocabulary projection, and gradient propagation through the model.

## Packaging

The repository includes `pyproject.toml` for Python project metadata and development dependencies.

Install the project with:

```bash
pip install -e .
```

Install development dependencies with:

```bash
pip install -e .[dev]
```

## Attribution

This implementation is based on the Transformer architecture presented in:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). *Attention Is All You Need*.

https://arxiv.org/abs/1706.03762

The project is independently implemented in PyTorch for educational, experimental, and portfolio purposes.
