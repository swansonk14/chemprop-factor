from typing import Callable

from torch.optim import Optimizer
from tqdm import trange

from data import MoleculeFactorDataset
from model import MatrixFactorizer


def train(model: MatrixFactorizer,
          data: MoleculeFactorDataset,
          loss_func: Callable,
          optimizer: Optimizer,
          batch_size: int = 50):
    model.train()

    data.shuffle()

    num_iters = len(data) // batch_size * batch_size

    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        model.zero_grad()

        batch = MoleculeFactorDataset(data[i:i + batch_size])
        mol_indices, task_indices, targets = batch.mol_indices(), batch.task_indices(), batch.targets()

        preds = model(mol_indices, task_indices)

        loss = loss_func(preds, targets)

        loss.backward()
        optimizer.step()
