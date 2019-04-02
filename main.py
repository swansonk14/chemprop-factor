from argparse import Namespace

from chemprop.data.utils import get_data
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

from data import convert_moleculedataset_to_moleculefactordataset
from evaluate import evaluate
from predict import predict, fill_matrix
from model import MatrixFactorizer
from parsing import parse_train_args
from train import train
from utils import split_data, save, load


def main(args: Namespace):
    print('Loading data')
    dataset = get_data(args.data_path)
    num_mols, num_tasks = len(dataset), dataset.num_tasks()
    args.num_mols, args.num_tasks = num_mols, num_tasks
    data = convert_moleculedataset_to_moleculefactordataset(dataset)

    print(f'Number of molecules = {num_mols:,}')
    print(f'Number of tasks = {num_tasks:,}')
    print(f'Number of known molecule-task pairs = {len(data):,}')

    print('Splitting data')
    train_data, val_data, test_data = split_data(data)

    print(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if args.checkpoint_path is not None:
        print('Loading saved model')
        model, loaded_args = load(args.checkpoint_path)
        assert args.num_mols == loaded_args.num_mols and args.num_tasks == loaded_args.num_tasks
        args.embedding_dim, args.hidden_dim, args.dropout_prob = loaded_args.embedding_dim, loaded_args.hidden_dim, loaded_args.dropout_prob
    else:
        print('Building new model')
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

    if args.save_path is not None:
        print('Saving model')
        save(model, args, args.save_path)
    
    if args.filled_matrix_path is not None:
        print('Filling matrix of data')
        fill_matrix(model, args, data)

if __name__ == '__main__':
    args = parse_train_args()
    main(args)
