import random
from typing import List, Tuple

from tqdm import trange

from model import MatrixFactorizer


def train(model: MatrixFactorizer,
          data: List[Tuple[int, int]],
          batch_size: int = 50):
    model.train()

    random.shuffle(data)

    num_iters = len(data) // batch_size * batch_size

    iter_size = batch_size

    for i in trange(0, num_iters, iter_size):
        model.zero_grad()

        batch = data[i:i + batch_size]
        mol_indices, task_indices = zip(*batch)
        preds = model(mol_indices, task_indices)


