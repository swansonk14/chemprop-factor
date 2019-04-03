from argparse import Namespace

from chemprop.data.utils import get_data
from chemprop.nn_utils import param_count
from chemprop.utils import get_metric_func, get_loss_func
from torch.optim import Adam
from tqdm import trange

from data import convert_moleculedataset_to_moleculefactordataset
from evaluate import evaluate
from predict import fill_matrix
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
        args.embedding_size, args.hidden_size, args.dropout, args.activation, args.dataset_type = \
            loaded_args.embedding_size, loaded_args.hidden_size, loaded_args.dropout, loaded_args.activation, loaded_args.dataset_type
        metric_func = get_metric_func(metric=args.metric)
    else:
        print('Building model')
        model = MatrixFactorizer(
            args,
            num_mols=num_mols,
            num_tasks=num_tasks,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
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
            batch_size=args.batch_size,
            random_mol_embeddings=args.random_mol_embeddings
        )
        val_score = evaluate(
            model=model,
            data=val_data,
            metric_func=metric_func,
            batch_size=args.batch_size,
            random_mol_embeddings=args.random_mol_embeddings
        )
        print(f'Validation {args.metric} = {val_score:.6f}')

    test_score = evaluate(
        model=model,
        data=test_data,
        metric_func=metric_func,
        batch_size=args.batch_size,
        random_mol_embeddings=args.random_mol_embeddings
    )
    print(f'Test {args.metric} = {test_score:.6f}')

    if args.save_path is not None:
        print('Saving model')
        save(model, args, args.save_path)
    
    if args.filled_matrix_path is not None:
        print('Filling matrix of data')
        fill_matrix(model, args, data)


if __name__ == '__main__':
    args = parse_train_args()
    main(args)
