from pathlib import Path

import torch
import torchmetrics
from torch.utils.data import DataLoader

from config import get_config
from dataset import BilingualDataset
from translate import greedy_decode


def evaluate_model(
    model,
    dataset,
    tokenizer_src,
    tokenizer_tgt,
    config,
    device,
    num_examples=None
):
    loader = DataLoader(
        BilingualDataset(
            dataset,
            tokenizer_src,
            tokenizer_tgt,
            config["lang_src"],
            config["lang_tgt"],
            config["seq_len"]
        ),
        batch_size=1,
        shuffle=False
    )

    expected = []
    predicted = []

    model.eval()

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if num_examples is not None and index >= num_examples:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                config["seq_len"],
                device
            )

            prediction = tokenizer_tgt.decode(
                output.detach().cpu().numpy(),
                skip_special_tokens=True
            )

            expected.append(batch["tgt_text"][0])
            predicted.append(prediction)

    if not predicted:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "bleu": 0.0,
            "expected": [],
            "predicted": []
        }

    cer = torchmetrics.CharErrorRate()(predicted, expected)
    wer = torchmetrics.WordErrorRate()(predicted, expected)
    bleu = torchmetrics.BLEUScore()(
        predicted,
        [[text] for text in expected]
    )

    return {
        "cer": cer.item(),
        "wer": wer.item(),
        "bleu": bleu.item(),
        "expected": expected,
        "predicted": predicted
    }


def load_validation_data(config, tokenizer_src, tokenizer_tgt):
    from datasets import load_dataset
    from torch.utils.data import random_split

    dataset = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train"
    )

    train_size = int(0.9 * len(dataset))
    _, validation_data = random_split(
        dataset,
        [
            train_size,
            len(dataset) - train_size
        ],
        generator=torch.Generator().manual_seed(
            config.get("seed", 42)
        )
    )

    return validation_data


if __name__ == "__main__":
    config = get_config()
    from translate import load_translation_model

    model, tokenizer_src, tokenizer_tgt, device = load_translation_model(config)
    validation_data = load_validation_data(
        config,
        tokenizer_src,
        tokenizer_tgt
    )

    metrics = evaluate_model(
        model,
        validation_data,
        tokenizer_src,
        tokenizer_tgt,
        config,
        device
    )

    output_path = Path(config["model_folder"]) / "evaluation.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "bleu": metrics["bleu"]
        },
        output_path
    )
