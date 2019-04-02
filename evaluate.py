from typing import List

from sklearn.metrics import roc_auc_score

from data import MoleculeFactorDataset
from model import MatrixFactorizer
from predict import predict


def evaluate_predictions(targets: List[int],
                         preds: List[float]):
    # TODO: support other metrics
    return roc_auc_score(targets, preds)


def evaluate(model: MatrixFactorizer,
             data: MoleculeFactorDataset,
             batch_size: int) -> float:
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size
    )

    targets = data.targets()

    score = evaluate_predictions(
        targets=targets,
        preds=preds
    )

    return score
