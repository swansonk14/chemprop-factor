from typing import List
from argparse import Namespace

import torch
from tqdm import trange

from data import MoleculeFactorDataset, MoleculeFactorDatapoint
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


def fill_matrix(model: MatrixFactorizer,
                args: Namespace, 
                data: MoleculeFactorDataset):
    predict_dataset = MoleculeFactorDataset([MoleculeFactorDatapoint(i, j, -1) for i in range(args.num_mols) for j in range(args.num_tasks)])
    preds = predict(model=model,
                    data=predict_dataset,
                    batch_size=args.batch_size)
    prediction_matrix = [[None for _ in range(args.num_tasks)] for _ in range(args.num_mols)]
    for datapoint, pred in zip(predict_dataset, preds):
        prediction_matrix[datapoint.mol_index][datapoint.task_index] = str(pred)
    # replace with original value where we have it
    for datapoint in data:
        prediction_matrix[datapoint.mol_index][datapoint.task_index] = str(datapoint.target)
    with open(args.data_path, 'r') as original, open(args.filled_matrix_path, 'w') as wf:
        header = original.readline()
        wf.write(header.strip() + '\n')
        mol_index = 0
        for line in original:
            smiles = line.strip().split(',')[0]
            wf.write(smiles + ',' + ','.join(prediction_matrix[mol_index]) + '\n')
            mol_index += 1
