from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    get_config,
    get_weights_file_path,
    latest_weights_file_path
)
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from tokenizer import get_or_build_tokenizer


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

    decoder_input = torch.full(
        (source.size(0), 1),
        sos_idx,
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

        if torch.all(next_word == eos_idx):
            break

    return decoder_input


def run_validation(
    model,
    validation_ds,
    tokenizer_tgt,
    max_len,
    device,
    global_step,
    writer,
    num_examples=100
):
    model.eval()

    expected = []
    predicted = []

    with torch.inference_mode():
        for count, batch in enumerate(validation_ds, start=1):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                max_len,
                device
            )

            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(
                model_out[0].detach().cpu().numpy(),
                skip_special_tokens=True
            )

            expected.append(target_text)
            predicted.append(model_out_text)

            if count == num_examples:
                break

    if not predicted:
        return {}

    cer = torchmetrics.CharErrorRate()(predicted, expected)
    wer = torchmetrics.WordErrorRate()(predicted, expected)
    bleu = torchmetrics.BLEUScore()(
        predicted,
        [[text] for text in expected]
    )

    if writer:
        writer.add_scalar(
            "validation/cer",
            cer.item(),
            global_step
        )
        writer.add_scalar(
            "validation/wer",
            wer.item(),
            global_step
        )
        writer.add_scalar(
            "validation/bleu",
            bleu.item(),
            global_step
        )
        writer.flush()

    return {
        "cer": cer.item(),
        "wer": wer.item(),
        "bleu": bleu.item()
    }


def get_ds(config):
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train"
    )

    tokenizer_src = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_src"]
    )

    tokenizer_tgt = get_or_build_tokenizer(
        config,
        ds_raw,
        config["lang_tgt"]
    )

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(
        ds_raw,
        [train_ds_size, val_ds_size],
        generator=torch.Generator().manual_seed(
            config["seed"]
        )
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"]
    )

    loader_kwargs = {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": config["num_workers"]
    }

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    return (
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt
    )


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_model(config):
    torch.manual_seed(config["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = get_device()

    model_folder = Path(
        f"{config['datasource']}_{config['model_folder']}"
    )
    model_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt
    ) = get_ds(config)

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(
        config["experiment_name"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]

    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
        if preload
        else None
    )

    if model_filename:
        state = torch.load(
            model_filename,
            map_location=device
        )

        model.load_state_dict(
            state["model_state_dict"]
        )
        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=config["label_smoothing"]
    ).to(device)

    for epoch in range(
        initial_epoch,
        config["num_epochs"]
    ):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model.train()

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Processing Epoch {epoch:02d}"
        )

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(
                encoder_input,
                encoder_mask
            )

            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )

            proj_output = model.project(
                decoder_output
            )

            loss = loss_fn(
                proj_output.reshape(
                    -1,
                    tokenizer_tgt.get_vocab_size()
                ),
                label.reshape(-1)
            )

            batch_iterator.set_postfix(
                loss=f"{loss.item():6.3f}"
            )

            writer.add_scalar(
                "train/loss",
                loss.item(),
                global_step
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["grad_clip"]
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(
            model,
            val_dataloader,
            tokenizer_tgt,
            config["seq_len"],
            device,
            global_step,
            writer,
            config["validation_batches"]
        )

        model_filename = get_weights_file_path(
            config,
            f"{epoch:02d}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "config": config
            },
            model_filename
        )

    writer.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model(get_config())
