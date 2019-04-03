from typing import List, Union
from argparse import Namespace

from chemprop.nn_utils import get_activation_function
from chemprop.models.mpn import MPN
import torch
import torch.nn as nn


class MatrixFactorizer(nn.Module):
    def __init__(self,
                 args: Namespace,
                 num_mols: int,
                 num_tasks: int,
                 embedding_size: int,
                 hidden_size: int,
                 dropout: float,
                 activation: str,
                 classification: bool):
        super(MatrixFactorizer, self).__init__()

        self.num_mols = num_mols
        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.classification = classification
        self.random_mol_embeddings = args.random_mol_embeddings

        if args.random_mol_embeddings:
            self.mol_embedding = nn.Embedding(self.num_mols, self.embedding_size)
        else:
            self.mol_embedding = MPN(args)
        self.task_embedding = nn.Embedding(self.num_tasks, self.embedding_size)
        self.W1 = nn.Linear(2 * self.embedding_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.act_func = get_activation_function(self.activation)
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, mol_indices: List[Union[int, str]], task_indices: List[int]) -> torch.FloatTensor:
        assert len(mol_indices) == len(task_indices)

        task_indices = torch.LongTensor(task_indices)
        if next(self.parameters()).is_cuda:
            task_indices = task_indices.cuda()

        if self.random_mol_embeddings:
            mol_indices = torch.LongTensor(mol_indices)
            if next(self.parameters()).is_cuda:
                mol_indices = mol_indices.cuda()

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
