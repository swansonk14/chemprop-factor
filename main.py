from argparse import ArgumentParser, Namespace

from chemprop.data.utils import get_data

from model import MatrixFactorizer


def main(args: Namespace):
    # Load data
    data = get_data(args.data_path)

    # Get number of molecules and tasks
    num_mols, num_tasks = len(data), data.num_tasks()

    # Construct model
    model = MatrixFactorizer(num_mols, num_tasks)

    # TODO


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    args = parser.parse_args()

    main(args)
