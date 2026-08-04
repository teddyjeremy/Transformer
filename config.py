from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 128,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch):
    model_folder = Path(
        f"{config['datasource']}_{config['model_folder']}"
    )
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = Path(
        f"{config['datasource']}_{config['model_folder']}"
    )

    if not model_folder.exists():
        return None

    weights_files = list(
        model_folder.glob(
            f"{config['model_basename']}*.pt"
        )
    )

    if not weights_files:
        return None

    return str(
        max(
            weights_files,
            key=lambda path: int(
                path.stem.replace(
                    config["model_basename"],
                    ""
                )
            )
        )
    )
