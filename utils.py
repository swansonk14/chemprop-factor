from argparse import Namespace
from typing import Tuple
import random

from data import MoleculeFactorDataset
from model import MatrixFactorizer

import torch


def split_data(data: MoleculeFactorDataset,
               args: Namespace,
               sizes: Tuple[int, int, int] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[MoleculeFactorDataset, MoleculeFactorDataset, MoleculeFactorDataset]:

    if args.num_real_tasks is not None:
        real_data, aux_data = \
            [d for d in data if d.task_index < args.num_real_tasks], \
            [d for d in data if d.task_index >= args.num_real_tasks]
        data = MoleculeFactorDataset(real_data)

    data.shuffle(seed)

    if args.split_type == 'entries':
        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]
    elif args.split_type == 'rows':
        mol_indices = list(range(args.num_mols))
        random.seed(seed)
        random.shuffle(mol_indices)

        train_size = int(sizes[0] * len(mol_indices))
        train_val_size = int((sizes[0] + sizes[1]) * len(mol_indices))
        train_mol_indices = mol_indices[:train_size]
        val_mol_indices = mol_indices[train_size:train_val_size]
        test_mol_indices = mol_indices[train_val_size:]
        
        train_data = [d for d in data if d.mol_index in train_mol_indices]
        val_data = [d for d in data if d.mol_index in val_mol_indices]
        test_data = [d for d in data if d.mol_index in test_mol_indices]
    else:
        raise ValueError(f'Split type "{args.split_type}" not supported.')

    if args.num_real_tasks is not None:
        train_data += aux_data

    return MoleculeFactorDataset(train_data), MoleculeFactorDataset(val_data), MoleculeFactorDataset(test_data)


def save(model: MatrixFactorizer, args: Namespace, path: str):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)


def load(path: str) -> Tuple[MatrixFactorizer, Namespace]:
    state = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = state['state_dict']
    args = state['args']
    model = MatrixFactorizer(
        num_mols=args.num_mols,
        num_tasks=args.num_tasks,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        activation=args.activation,
        classification=(args.dataset_type == 'classification')
    )
    model.load_state_dict(state_dict)

    return model, args
