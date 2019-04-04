from typing import Callable, List

from data import MoleculeFactorDataset
from model import MatrixFactorizer
from predict import predict

from chemprop.data.scaler import StandardScaler


def evaluate_predictions(targets: List[int],
                         preds: List[float],
                         num_tasks: int,
                         task_indices: List[int],
                         metric_func: Callable) -> List[float]:
    targets_by_task = [[] for _ in range(num_tasks)]
    preds_by_task = [[] for _ in range(num_tasks)]

    for target, pred, task_index in zip(targets, preds, task_indices):
        targets_by_task[task_index].append(target)
        preds_by_task[task_index].append(pred)

    import pdb; pdb.set_trace()

    results = []
    for task_targets, task_preds in zip(targets_by_task, preds_by_task):
        results.append(metric_func(task_targets, task_preds))

    return results


def evaluate(model: MatrixFactorizer,
             data: MoleculeFactorDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             scaler: StandardScaler = None,
             random_mol_embeddings: bool = False) -> List[float]:
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler,
        random_mol_embeddings=random_mol_embeddings
    )

    targets = data.targets()
    task_indices = data.task_indices()

    score = evaluate_predictions(
        targets=targets,
        preds=preds,
        num_tasks=num_tasks,
        task_indices=task_indices,
        metric_func=metric_func
    )

    return score
