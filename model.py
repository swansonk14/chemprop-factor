from typing import List

from chemprop.nn_utils import get_activation_function
import torch
import torch.nn as nn


class MatrixFactorizer(nn.Module):
    def __init__(self,
                 num_mols: int,
                 num_tasks: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 activation: str,
                 classification: bool):
        super(MatrixFactorizer, self).__init__()

        self.num_mols = num_mols
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.classification = classification

        self.mol_embedding = nn.Embedding(self.num_mols, self.embedding_dim)
        self.task_embedding = nn.Embedding(self.num_tasks, self.embedding_dim)
        self.W1 = nn.Linear(2 * self.embedding_dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.act_func = get_activation_function(self.activation)
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, mol_indices: List[int], task_indices: List[int]) -> torch.FloatTensor:
        assert len(mol_indices) == len(task_indices)

        mol_indices, task_indices = torch.LongTensor(mol_indices), torch.LongTensor(task_indices)

        if next(self.parameters()).is_cuda:
            mol_indices, task_indices = mol_indices.cuda(), task_indices.cuda()

        # Look up molecule and task embeddings
        mol_embeddings, task_embeddings = self.mol_embedding(mol_indices), self.task_embedding(task_indices)

        # Concatenate molecule and task embeddings
        joint_embeddings = torch.cat((mol_embeddings, task_embeddings), dim=1)
        joint_embeddings = self.dropout_layer(joint_embeddings)

        # Run neural network
        hiddens = self.act_func(self.W1(joint_embeddings))
        hiddens = self.dropout_layer(hiddens)
        output = self.W2(hiddens)
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        output = output.squeeze(dim=-1)

        return output
