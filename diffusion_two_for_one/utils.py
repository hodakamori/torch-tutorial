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


def add_diffusion_noise(coordinates, noise_level, noise_multiplier=0.1):
    alpha_list = torch.Tensor([1 - noise_multiplier] * noise_level)
    alpha_bar = torch.prod(alpha_list)
    noised_coordinates = torch.sqrt(alpha_bar) * coordinates + torch.sqrt(
        1 - alpha_bar
    ) * torch.randn_like(coordinates)
    return noised_coordinates
