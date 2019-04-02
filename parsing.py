from argparse import ArgumentParser, Namespace

import torch


def add_train_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--save_path', type=str,
                        help='Path to save checkpoint at the end')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to load checkpoint from; skips training')
    parser.add_argument('--filled_matrix_path', type=str,
                        help='Path to save the data matrix filled with new predictions')

    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Molecule/task embedding dimensionality')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='Neural network dimensionality')
    parser.add_argument('--dropout_prob', type=float, default=0.05,
                        help='Dropout probability')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')


def modify_train_args(args: Namespace):
    args.cuda = torch.cuda.is_available()


def parse_train_args() -> Namespace:
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
