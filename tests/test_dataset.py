import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from dataset import BilingualDataset, causal_mask


def build_test_tokenizer(sentences):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=1
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    return tokenizer


def test_dataset_masks_and_lengths():
    source_tokenizer = build_test_tokenizer(["hello world"])
    target_tokenizer = build_test_tokenizer(["bonjour monde"])

    dataset = BilingualDataset(
        [
            {
                "translation": {
                    "en": "hello world",
                    "fr": "bonjour monde"
                }
            }
        ],
        source_tokenizer,
        target_tokenizer,
        "en",
        "fr",
        8
    )

    item = dataset[0]

    assert item["encoder_input"].shape == (8,)
    assert item["decoder_input"].shape == (8,)
    assert item["label"].shape == (8,)
    assert item["encoder_mask"].shape == (1, 1, 8)
    assert item["decoder_mask"].shape == (1, 8, 8)
    assert torch.equal(
        item["decoder_mask"],
        item["decoder_mask"] & causal_mask(8)
    )


def test_dataset_rejects_long_sequences():
    source_tokenizer = build_test_tokenizer(["one two three four five"])
    target_tokenizer = build_test_tokenizer(["un deux trois quatre cinq"])

    dataset = BilingualDataset(
        [
            {
                "translation": {
                    "en": "one two three four five",
                    "fr": "un deux trois quatre cinq"
                }
            }
        ],
        source_tokenizer,
        target_tokenizer,
        "en",
        "fr",
        4
    )

    try:
        dataset[0]
    except ValueError as error:
        assert str(error) == "Sentence is too long"
    else:
        raise AssertionError("Expected ValueError for an oversized sequence")
