from typing import List

import torch
from tqdm import trange

from data import MoleculeFactorDataset
from model import MatrixFactorizer


def predict(model: MatrixFactorizer,
            data: MoleculeFactorDataset,
            batch_size: int) -> List[float]:
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        batch = MoleculeFactorDataset(data[i:i + batch_size])
        mol_indices, task_indices = batch.mol_indices(), batch.task_indices()

        # Run model
        with torch.no_grad():
            batch_preds = model(mol_indices, task_indices)

        # Collect predictions
        batch_preds = batch_preds.data.cpu().numpy().tolist()
        preds.extend(batch_preds)

    return preds
