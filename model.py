from typing import List

import torch
import torch.nn as nn


class MatrixFactorizer(nn.Module):
    def __init__(self,
                 num_mols: int,
                 num_tasks: int,
                 embedding_dim: int = 10,
                 p: float = 0.05):
        super(MatrixFactorizer, self).__init__()

        self.num_mols = num_mols
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.p = p

        self.mol_embedding = nn.Embedding(self.num_mols, self.embedding_dim)
        self.task_embedding = nn.Embedding(self.num_tasks, self.embedding_dim)
        self.W1 = nn.Linear()
        self.W2 = nn.Linear()
        self.dropout = nn.Dropout(self.p)
        self.relu = nn.ReLU()

    def forward(self, mols: List[int]) -> torch.FloatTensor:
        pass
