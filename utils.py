from typing import Tuple

from data import MoleculeFactorDataset


def split_data(data: MoleculeFactorDataset,
               sizes: Tuple[int, int, int] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[MoleculeFactorDataset, MoleculeFactorDataset, MoleculeFactorDataset]:
    data.shuffle(seed)

    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_val_size]
    test_data = data[train_val_size:]

    return MoleculeFactorDataset(train_data), MoleculeFactorDataset(val_data), MoleculeFactorDataset(test_data)
