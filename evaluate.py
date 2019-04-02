from typing import List

from sklearn.metrics import roc_auc_score

from data import MoleculeFactorDataset
from model import MatrixFactorizer
from predict import predict


def evaluate_predictions(preds: List[float],
                         targets: List[int]):
    # TODO: support other metrics
    return roc_auc_score(preds, targets)


def evaluate(model: MatrixFactorizer,
             data: MoleculeFactorDataset) -> float:
    preds = predict(
        model=model,
        data=data
    )

    targets = data.targets()

    score = evaluate_predictions(
        preds=preds,
        targets=targets
    )

    return score
