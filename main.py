from argparse import ArgumentParser, Namespace

from chemprop.data.utils import get_data
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from data import convert_moleculedataset_to_moleculefactordataset
from evaluate import evaluate
from model import MatrixFactorizer
from train import train
from utils import split_data


def main(args: Namespace):
    print('Loading data')
    dataset = get_data(args.data_path)
    num_mols, num_tasks = len(dataset), dataset.num_tasks()
    data = convert_moleculedataset_to_moleculefactordataset(dataset)

    print(f'Number of molecules = {num_mols:,}')
    print(f'Number of tasks = {num_tasks:,}')
    print(f'Number of known molecule-task pairs = {len(data):,}')

    print('Splitting data')
    train_data, val_data, test_data = split_data(data)

    print(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    print('Building model')
    model = MatrixFactorizer(
        num_mols=num_mols,
        num_tasks=num_tasks,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout_prob
    )
    loss_func = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    print('Training')
    for epoch in trange(args.epochs):
        print(f'Epoch {epoch}')
        train(
            model=model,
            data=train_data,
            loss_func=loss_func,
            optimizer=optimizer,
            batch_size=args.batch_size
        )
        val_score = evaluate(
            model=model,
            data=val_data,
            batch_size=args.batch_size
        )
        print(f'Validation auc = {val_score:.6f}')

    test_score = evaluate(
        model=model,
        data=test_data,
        batch_size=args.batch_size
    )
    print(f'Test auc = {test_score:.6f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')

    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Molecule/task embedding dimensionality')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='Neural network dimensionality')
    parser.add_argument('--dropout_prob', type=float, default=0.05,
                        help='Dropout probability')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    main(args)
