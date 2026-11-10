"""The five files needed by inference, with immutable download URLs."""
import hashlib
import json
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name("weights.json").read_text())
DIT, TEXT, VAE, TOKENIZER, VAE_CONFIG = (entry["path"] for entry in MANIFEST["files"])


def check_models(directory, *, hashes=False):
    directory = Path(directory).expanduser().resolve()
    for entry in MANIFEST["files"]:
        path = directory / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}. Run h3-image-mlx fetch --models '{directory}' to complete this directory.")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Unexpected file size: {path}")
        if hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return directory


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
