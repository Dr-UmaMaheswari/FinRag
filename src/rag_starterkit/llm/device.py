import torch
import os

def get_torch_device() -> str:
    # Allow manual override if needed
    forced = os.getenv("TORCH_DEVICE")
    if forced:
        return forced

    return "cuda" if torch.cuda.is_available() else "cpu"
