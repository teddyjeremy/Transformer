from pathlib import Path
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = {
    "source_lang": "en",
    "target_lang": "fr",
    "tokenizer_file": "tokenizer_{0}.json"
}


# --------------------------------------------------
# Get all sentences from the dataset
# --------------------------------------------------

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


# --------------------------------------------------
# Build tokenizer
# --------------------------------------------------

def build_tokenizer(config, dataset, lang):
    
    tokenizer_path = Path(
        config["tokenizer_file"].format(lang)
    )

    # If tokenizer doesn't already exist, create it
    if not tokenizer_path.exists():

        tokenizer = Tokenizer(
            WordLevel(unk_token="[UNK]")
        )

        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=[
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "[SOS]"
            ],
            min_frequency=2
        )

        tokenizer.train_from_iterator(
            get_all_sentences(dataset, lang),
            trainer=trainer
        )

        tokenizer.save(str(tokenizer_path))

    # Otherwise load existing tokenizer
    else:

        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

    return tokenizer


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def get_dataset(config):

    dataset = load_dataset(
        "opus_books",
        f"{config['source_lang']}-{config['target_lang']}",
        split="train"
    )

    return dataset


# --------------------------------------------------
# Get raw dataset
# --------------------------------------------------

ds_raw = get_dataset(config)


# --------------------------------------------------
# Build source tokenizer
# --------------------------------------------------

tokenizer_src = build_tokenizer(
    config,
    ds_raw,
    config["source_lang"]
)


# --------------------------------------------------
# Build target tokenizer
# --------------------------------------------------

tokenizer_tgt = build_tokenizer(
    config,
    ds_raw,
    config["target_lang"]
)


train_dataset_size = len(ds_raw)
val_dataset_size = len(ds_raw) - train_dataset_size

train_ds_raw, val_ds_raw = random_split(
    ds_raw, 
    [train_dataset_size, val_dataset_size]
)

