from argparse import ArgumentParser, Namespace
import os
import pickle

from chemprop.features import get_available_features_generators, get_features_generator

def save_features(args: Namespace):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    all_features = []
    for path in args.features_path:
        with open(path, 'rb') as f:
            all_features.append(pickle.load(f).todense())
    with open(args.data_path, 'r') as rf, open(args.save_path, 'w') as wf:
        header = rf.readline().strip()
        num_aux_features = sum([features.shape[1] for features in all_features])
        for i in range(num_aux_features):
            header += f',auxiliary_feature_{i}'
        wf.write(header + '\n')
        index = 0
        for line in rf:
            wf.write(line.strip())
            for features in all_features:
                wf.write(',' + ','.join([str(features[i, j]) for j in range(features.shape[1])]))
            wf.write('\n')
            index += 1
        
        # ensure we did the alignment right
        for features in all_features:
            assert index == len(features)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use as auxiliary labels in matrix')  
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to augmented data file')
    args = parser.parse_args()

    save_features(args)
