import random
from typing import List, Union

from chemprop.data import MoleculeDataset


class MoleculeFactorDatapoint:
    def __init__(self, mol_index: int, smiles: str, task_index: int, target: int):
        self.mol_index = mol_index
        self.smiles = smiles
        self.task_index = task_index
        self.target = target


class MoleculeFactorDataset:
    def __init__(self, data: List[MoleculeFactorDatapoint]):
        self.data = data

    def mol_indices(self) -> List[int]:
        return [d.mol_index for d in self.data]

    def smiles(self) -> List[str]:
        return [d.smiles for d in self.data]

    def task_indices(self) -> List[int]:
        return [d.task_index for d in self.data]

    def targets(self) -> List[int]:
        return [d.target for d in self.data]
    
    def set_targets(self, targets: List[List[float]]):
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].target = targets[i]

    def shuffle(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeFactorDatapoint, List[MoleculeFactorDatapoint]]:
        return self.data[item]


def convert_moleculedataset_to_moleculefactordataset(dataset: MoleculeDataset) -> MoleculeFactorDataset:
    datapoints = []

    for mol_index, datapoint in enumerate(dataset):
        for task_index, target in enumerate(datapoint.targets):
            if target is not None:
                datapoints.append(MoleculeFactorDatapoint(mol_index, datapoint.smiles, task_index, target))

    return MoleculeFactorDataset(datapoints)
