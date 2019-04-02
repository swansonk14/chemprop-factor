from argparse import ArgumentParser, Namespace

from chemprop.data.utils import get_data
import numpy as np
from tqdm import trange

from data import convert_moleculedataset_to_moleculefactordataset
from evaluate import evaluate
from model import MatrixFactorizer
from train import train
from utils import split_data


def main(args: Namespace):
    # Load data
    dataset = get_data(args.data_path)
    num_mols, num_tasks = len(dataset), dataset.num_tasks()
    data = convert_moleculedataset_to_moleculefactordataset(dataset)

    # Split data
    train_data, val_data, test_data = split_data(data)

    # Construct model
    model = MatrixFactorizer(num_mols, num_tasks)

    # Run training
    for epoch in trange(args.epochs):
        print(f'Epoch {epoch}')
        train(
            model=model,
            data=train_data
        )
        val_scores = evaluate(
            model=model,
            data=val_data
        )
        print(f'Validation score = {np.mean(val_scores):.6f}')

    # Test
    test_scores = evaluate(
        model=model,
        data=test_data
    )
    print(f'Test score = {np.mean(test_scores):.6f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    args = parser.parse_args()

    main(args)
