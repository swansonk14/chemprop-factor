from argparse import ArgumentParser, Namespace

from chemprop.data.utils import get_data
import torch.nn as nn
from torch.optim import Adam
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
    loss_func = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Run training
    for epoch in trange(args.epochs):
        print(f'Epoch {epoch}')
        train(
            model=model,
            data=train_data,
            loss_func=loss_func,
            optimizer=optimizer
        )
        val_score = evaluate(
            model=model,
            data=val_data
        )
        print(f'Validation score = {val_score:.6f}')

    # Test
    test_score = evaluate(
        model=model,
        data=test_data
    )
    print(f'Test score = {test_score:.6f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    args = parser.parse_args()

    main(args)
