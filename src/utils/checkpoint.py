from pathlib import Path
import torch


def save_checkpoint(path, state):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(p))


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
