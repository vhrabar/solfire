import argparse
import os
import random

import numpy as np
import torch

from data import load_annotations, train_val_split, FilamentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, _ = load_annotations()
    train_ids, val_ids = train_val_split(images, val_fraction=0.2)
    train_dataset = FilamentDataset(train_ids, size=512, category_ids=(1, 2))
    valid_dataset = FilamentDataset(val_ids, size=512, category_ids=(1, 2))




def set_seeds(seed: int) -> None:
    """
    Sets random seeds for reproducibility
    :param seed: Seed value
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seeds(1)
    main()