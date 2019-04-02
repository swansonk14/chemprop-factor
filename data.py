from typing import List

class MoleculeFactorDatapoint:
    def __init__(self, mol_index: int, targets: List[int]):
        self.mol_index = int
        self.targets = targets

    def num_tasks(self) -> int:
        return len(self.targets)

class MoleculeFactorDataset:
    def __init__(self, data: List[MoleculeFactorDatapoint]):
        self.data = data

    def mol_indices(self) -> List[int]:
        return [d.mol_index for d in self.data]

    def task