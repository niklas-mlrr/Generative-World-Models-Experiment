import os
import random

import numpy as np
import torch


def configure_runtime() -> None:
    current = os.environ.get("MPLCONFIGDIR", "")
    if current and os.path.isdir(current) and os.access(current, os.W_OK):
        return
    mpl_dir = "/tmp/mplconfig_reliability_paradox"
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_device(raw: str) -> torch.device:
    s = raw.strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    raise ValueError("--wm-device must be 'mps' or 'cpu'")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
