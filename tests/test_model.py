import torch

from dataset import causal_mask
from model import MultiHeadAttentionBlock, build_transformer


def test_causal_mask():
    mask = causal_mask(4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ]
    ).unsqueeze(0)

    assert torch.equal(mask, expected)


def test_attention_mask_blocks_future_positions():
    torch.manual_seed(42)

    attention = MultiHeadAttentionBlock(
        d_model=16,
        h=4,
        dropout=0.0
    )

    query = torch.randn(1, 4, 16)
    key = torch.randn(1, 4, 16)
    value = torch.randn(1, 4, 16)
    mask = causal_mask(4).unsqueeze(0)

    _, scores = attention.attention(
        query.view(1, 4, 4, 4).transpose(1, 2),
        key.view(1, 4, 4, 4).transpose(1, 2),
        value.view(1, 4, 4, 4).transpose(1, 2),
        mask,
        None
    )

    assert torch.all(scores.triu(diagonal=1) == 0)


def test_transformer_shapes():
    src_vocab_size = 100
    tgt_vocab_size = 120
    seq_len = 16
    batch_size = 2

    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        seq_len,
        seq_len,
        d_model=128,
        N=2,
        h=8,
        dropout=0.0,
        d_ff=512
    )

    source = torch.randint(
        0,
        src_vocab_size,
        (batch_size, seq_len)
    )

    target = torch.randint(
        0,
        tgt_vocab_size,
        (batch_size, seq_len)
    )

    source_mask = torch.ones(
        batch_size,
        1,
        1,
        seq_len,
        dtype=torch.bool
    )

    target_mask = causal_mask(
        seq_len
    ).unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
        -1
    )

    encoder_output = model.encode(
        source,
        source_mask
    )

    decoder_output = model.decode(
        encoder_output,
        source_mask,
        target,
        target_mask
    )

    projection = model.project(
        decoder_output
    )

    assert encoder_output.shape == (
        batch_size,
        seq_len,
        128
    )

    assert decoder_output.shape == (
        batch_size,
        seq_len,
        128
    )

    assert projection.shape == (
        batch_size,
        seq_len,
        tgt_vocab_size
    )


def test_transformer_backward():
    torch.manual_seed(42)

    model = build_transformer(
        32,
        40,
        8,
        8,
        d_model=64,
        N=2,
        h=8,
        dropout=0.0,
        d_ff=128
    )

    source = torch.randint(0, 32, (2, 8))
    target = torch.randint(0, 40, (2, 8))
    source_mask = torch.ones(2, 1, 1, 8, dtype=torch.bool)
    target_mask = causal_mask(8).unsqueeze(0).expand(2, -1, -1, -1)

    encoder_output = model.encode(source, source_mask)
    decoder_output = model.decode(
        encoder_output,
        source_mask,
        target,
        target_mask
    )

    logits = model.project(decoder_output)
    loss = logits.mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(gradient is not None for gradient in gradients)
