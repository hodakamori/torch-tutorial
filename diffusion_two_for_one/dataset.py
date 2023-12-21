import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import MDAnalysis as mda


class CGCoordsDataset(Dataset):
    def __init__(self, topology_path, traj_path) -> None:
        super().__init__()
        u = mda.Universe(topology_path, traj_path)
        self.bonds = torch.Tensor([tuple(int(i.ix) for i in bond) for bond in u.bonds])
        self.u = u

    def __getitem__(self, index: int) -> torch.Tensor:
        for tr in self.u.trajectory[index]:
            positions = self.u.atoms.positions
        indices = [i for i in range(len(positions))]
        return torch.tensor(indices, dtype=torch.long), torch.Tensor(positions)

    def __len__(self) -> int:
        return len(self.u.trajectory)
