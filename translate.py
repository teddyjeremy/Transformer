from pathlib import Path

import torch

from config import (
    get_config,
    get_weights_file_path,
    latest_weights_file_path
)
from dataset import causal_mask
from model import build_transformer
from tokenizers import Tokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_tokenizers(config):
    source_path = Path(
        config["tokenizer_file"].format(config["lang_src"])
    )
    target_path = Path(
        config["tokenizer_file"].format(config["lang_tgt"])
    )

    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "Tokenizer files were not found. Run train.py first."
        )

    return (
        Tokenizer.from_file(str(source_path)),
        Tokenizer.from_file(str(target_path))
    )


def load_model(
    config,
    tokenizer_src,
    tokenizer_tgt,
    device=None,
    checkpoint="latest"
):
    device = device or get_device()

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    ).to(device)

    model_path = (
        latest_weights_file_path(config)
        if checkpoint == "latest"
        else get_weights_file_path(config, checkpoint)
    )

    if not model_path:
        raise FileNotFoundError(
            "No model checkpoint was found. Train the model first."
        )

    state = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        state["model_state_dict"]
    )
    model.eval()

    return model


def encode_source(text, tokenizer_src, seq_len):
    source_tokens = tokenizer_src.encode(text).ids
    padding_length = seq_len - len(source_tokens) - 2

    if padding_length < 0:
        raise ValueError("Source sentence is too long")

    sos = tokenizer_src.token_to_id("[SOS]")
    eos = tokenizer_src.token_to_id("[EOS]")
    pad = tokenizer_src.token_to_id("[PAD]")

    source = torch.tensor(
        [
            sos,
            *source_tokens,
            eos,
            *([pad] * padding_length)
        ],
        dtype=torch.long
    ).unsqueeze(0)

    source_mask = (
        (source != pad)
        .unsqueeze(1)
        .unsqueeze(1)
    )

    return source, source_mask


def greedy_decode(
    model,
    source,
    source_mask,
    tokenizer_tgt,
    max_len,
    device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(
        source,
        source_mask
    )

    decoder_input = torch.tensor(
        [[sos_idx]],
        dtype=source.dtype,
        device=device
    )

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(
            decoder_input.size(1)
        ).to(device)

        decoder_output = model.decode(
            encoder_output,
            source_mask,
            decoder_input,
            decoder_mask
        )

        probabilities = model.project(
            decoder_output[:, -1]
        )

        next_word = probabilities.argmax(
            dim=-1,
            keepdim=True
        )

        decoder_input = torch.cat(
            [decoder_input, next_word],
            dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def translate_text(
    text,
    model,
    tokenizer_src,
    tokenizer_tgt,
    config,
    device=None
):
    device = device or next(model.parameters()).device

    source, source_mask = encode_source(
        text,
        tokenizer_src,
        config["seq_len"]
    )

    source = source.to(device)
    source_mask = source_mask.to(device)

    with torch.inference_mode():
        output = greedy_decode(
            model,
            source,
            source_mask,
            tokenizer_tgt,
            config["seq_len"],
            device
        )

    return tokenizer_tgt.decode(
        output.detach().cpu().numpy(),
        skip_special_tokens=True
    )


def load_translation_model(config=None, checkpoint="latest"):
    config = config or get_config()
    device = get_device()
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    model = load_model(
        config,
        tokenizer_src,
        tokenizer_tgt,
        device,
        checkpoint
    )

    return model, tokenizer_src, tokenizer_tgt, device
