from argparse import ArgumentParser, Namespace

from chemprop.data.utils import get_data
import numpy as np
from tqdm import trange

from evaluate import evaluate
from model import MatrixFactorizer
from train import train
from utils import split_data


def main(args: Namespace):
    # Load data
    data = get_data(args.data_path)
    targets = data.targets()

    # Get number of molecules and tasks
    num_mols, num_tasks = len(data), data.num_tasks()

    # Determine known molecule-task pairs
    known_pairs = [(mol_index, task_index) for mol_index in range(num_mols) for task_index in range(num_tasks) if targets[mol_index][task_index] is not None]

    # Split data
    train_data, val_data, test_data = split_data(known_pairs)

    # Construct model
    model = MatrixFactorizer(num_mols, num_tasks)

    # TODO
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
