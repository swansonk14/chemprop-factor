from typing import Callable, List

from data import MoleculeFactorDataset
from model import MatrixFactorizer
from predict import predict


def evaluate_predictions(targets: List[int],
                         preds: List[float],
                         metric_func: Callable):
    return metric_func(targets, preds)


def evaluate(model: MatrixFactorizer,
             data: MoleculeFactorDataset,
             metric_func: Callable,
             batch_size: int) -> float:
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size
    )

    targets = data.targets()

    score = evaluate_predictions(
        targets=targets,
        preds=preds,
        metric_func=metric_func
    )

    return score
