# Transformer English to French

A from-scratch PyTorch implementation of the original encoder-decoder Transformer for English-to-French neural machine translation using OPUS Books.

The project is based on the architecture introduced by Vaswani et al. in *Attention Is All You Need* and uses the `hkproj/pytorch-transformer` implementation as a reference while extending the training, evaluation, inference, testing, and notebook workflow.

Paper: https://arxiv.org/abs/1706.03762

Reference implementation: https://github.com/hkproj/pytorch-transformer

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
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   └── Transformer_English_to_French.ipynb
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

The encoder receives a padding mask with shape `(batch, 1, 1, source_length)`.

The decoder receives a combined padding and causal mask with shape `(batch, 1, target_length, target_length)` after batching. The causal mask prevents a target position from attending to future positions during teacher forcing.

### Embeddings

Token embeddings are scaled by `sqrt(d_model)`. Sinusoidal positional encodings are registered as buffers and added before the encoder or decoder stack.

### Multi-head attention

The implementation contains independent query, key, and value projections, head splitting, scaled dot-product attention, masking, dropout, head concatenation, and output projection.

For each head:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

The implementation also retains attention probabilities on each attention block for inspection during experiments.

### Encoder

Each encoder block contains:

1. Multi-head self-attention
2. Residual connection with layer normalization and dropout
3. Position-wise feed-forward network
4. Residual connection with layer normalization and dropout

Six blocks are used by default.

### Decoder

Each decoder block contains:

1. Masked multi-head self-attention
2. Residual connection with layer normalization and dropout
3. Encoder-decoder cross-attention
4. Residual connection with layer normalization and dropout
5. Position-wise feed-forward network
6. Residual connection with layer normalization and dropout

Six blocks are used by default.

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

The training pipeline handles tokenizers, deterministic train-validation splitting, DataLoaders, model construction, AdamW optimization, gradient clipping, label smoothing, TensorBoard logging, validation metrics, and checkpoint persistence.

Default model settings are six encoder layers, six decoder layers, eight attention heads, `d_model=512`, `d_ff=2048`, dropout `0.1`, and sequence length `128`.

## Checkpointing

Checkpoints are written under the configured `opus_books_weights` directory and excluded from version control.

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

The evaluation pipeline supports:

- Character Error Rate
- Word Error Rate
- BLEU
- Greedy autoregressive decoding
- Validation-set evaluation

Evaluation results are stored separately from model checkpoints.

## Translation

Load a trained model and translate an English sentence:

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

Training loss and validation CER, WER, and BLEU are recorded during training.

## Notebook

`notebooks/Transformer_English_to_French.ipynb` provides an implementation-level walkthrough of the project.

The notebook covers:

1. Configuration and reproducibility
2. OPUS Books loading
3. WordLevel tokenization
4. Source and target sequence construction
5. Padding masks
6. Causal masking
7. Token embeddings
8. Sinusoidal positional encoding
9. Layer normalization
10. Feed-forward networks
11. Scaled dot-product attention
12. Multi-head attention
13. Encoder blocks
14. Decoder blocks
15. Cross-attention
16. Transformer assembly
17. Tensor-shape inspection
18. Cross-entropy training objective
19. Gradient clipping
20. Autoregressive greedy decoding
21. Attention inspection
22. Checkpoint serialization

The notebook contains the core architecture implementation directly rather than only importing the project model.

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover causal-mask correctness and encoder, decoder, and projection tensor shapes.

## Attribution

This implementation is based on the Transformer architecture presented in:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). *Attention Is All You Need*.

https://arxiv.org/abs/1706.03762

The implementation structure and training approach were also informed by:

https://github.com/hkproj/pytorch-transformer

This repository is an independent implementation and extension rather than a copy of that repository.
