import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch


def smiles2structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    conf = mol.GetConformer(0)

    coordinates = []
    for i, (x, y, z) in enumerate(conf.GetPositions()):
        coordinates.append([x, y, z])
    coordinates = np.array(coordinates)

    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])

    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append((i, j))
    bonds = np.array(bonds)

    return (
        torch.tensor(coordinates, dtype=torch.float32, requires_grad=True),
        atomic_symbols,
        bonds,
    )


def add_diffusion_noise(coordinates, noise_levels, max_noise_level=25):
    beta = torch.linspace(0.0001, 0.02, max_noise_level)
    alpha = 1 - beta
    alpha_bars = torch.cumprod(alpha, dim=0)
    alpha_bar = torch.zeros_like(noise_levels, dtype=torch.float32)
    for i in range(noise_levels.shape[0]):
        for j in range(noise_levels.shape[1]):
            for k in range(noise_levels.shape[2]):
                alpha_bar[i, j, k] = alpha_bars[noise_levels[i, j, k] - 1]
    noised_coordinates = torch.sqrt(alpha_bar) * coordinates + torch.sqrt(
        1 - alpha_bar
    ) * torch.randn_like(coordinates)

    return noised_coordinates
