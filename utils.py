import random
from typing import List, Tuple


def split_data(data: List[Tuple[int, int]],
               sizes: Tuple[int, int, int] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    random.seed(seed)
    random.shuffle(data)

    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    train = data[:train_size]
    val = data[train_size:train_val_size]
    test = data[train_val_size:]

    return train, val, test
