from typing import Tuple

from data import MoleculeFactorDataset
from model import MatrixFactorizer

import torch


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

def save(model, args, path):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)

def load(path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = state['state_dict']
    args = state['args']
    model = MatrixFactorizer(
        num_mols=args.num_mols,
        num_tasks=args.num_tasks,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout_prob
    )
    model.load_state_dict(state_dict)
    return model, args