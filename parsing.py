from argparse import ArgumentParser, Namespace
from chemprop.parsing import add_train_args as chemprop_add_train_args, modify_train_args as chemprop_modify_train_args

import torch


def add_train_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['classification', 'regression'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--metric', type=str,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'r2', 'accuracy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--save_path', type=str,
                        help='Path to save checkpoint at the end')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to load checkpoint from; skips training')
    parser.add_argument('--filled_matrix_path', type=str,
                        help='Path to save the data matrix filled with new predictions')

    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Molecule/task embedding dimensionality')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Neural network dimensionality')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout probability')
    parser.add_argument('--mpn', action='store_true', default=False,
                        help='Use mpn for each molecule instead of random embeddings')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')


def modify_train_args(args: Namespace):
    args.cuda = torch.cuda.is_available()

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'r2'])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')
    
    args.random_mol_embeddings = not args.mpn


def parse_train_args() -> Namespace:
    parser = ArgumentParser(conflict_handler='resolve')
    chemprop_add_train_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    chemprop_modify_train_args(args)
    modify_train_args(args)

    return args
