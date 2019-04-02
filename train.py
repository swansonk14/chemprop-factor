from tqdm import trange

from data import MoleculeFactorDataset
from model import MatrixFactorizer


def train(model: MatrixFactorizer,
          data: MoleculeFactorDataset,
          batch_size: int = 50):
    model.train()

    data.shuffle()

    num_iters = len(data) // batch_size * batch_size

    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        model.zero_grad()

        batch = data[i:i + batch_size]
        mol_indices, task_indices = zip(*batch)
        preds = model(mol_indices, task_indices)


