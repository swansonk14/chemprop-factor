from argparse import Namespace

from chemprop.data.utils import get_data
from chemprop.nn_utils import param_count
from chemprop.utils import get_metric_func, get_loss_func
from torch.optim import Adam
from tqdm import trange

from data import convert_moleculedataset_to_moleculefactordataset
from evaluate import evaluate
from model import MatrixFactorizer
from parsing import parse_train_args
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
        dropout=args.dropout,
        activation=args.activation,
        classification=(args.dataset_type == 'classification')
    )
    print(model)
    print(f'Number of parameters = {param_count(model):,}')

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)
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
            metric_func=metric_func,
            batch_size=args.batch_size
        )
        print(f'Validation {args.metric} = {val_score:.6f}')

    test_score = evaluate(
        model=model,
        data=test_data,
        metric_func=metric_func,
        batch_size=args.batch_size
    )
    print(f'Test {args.metric} = {test_score:.6f}')


if __name__ == '__main__':
    args = parse_train_args()
    main(args)
