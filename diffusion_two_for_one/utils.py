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
    for i, (x,y,z) in enumerate(conf.GetPositions()):
       coordinates.append([x, y, z])
    coordinates = np.array(coordinates)

    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])

    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append((i,j))
    bonds = np.array(bonds)

    return torch.tensor(coordinates, dtype=torch.float32, requires_grad=True), atomic_symbols, bonds

def get_gradients(
        coordinates,
        energy
        ):
    num_atoms = energy.shape[1]
    gradients = []
    for i in range(num_atoms):
        if coordinates.grad is not None:
            coordinates.grad.zero_()
        energy[0][i].backward(retain_graph=True)
        gradients.append(coordinates.grad[i].clone())
    gradients = torch.stack(gradients, dim=0)
    return gradients

def add_diffusion_noise(
        coordinates,
        noise_level,
        noise_multiplier=0.1
        ):
    for _ in range(noise_level):
        noise = torch.randn_like(coordinates) * noise_multiplier
        coordinates = coordinates + noise
    return coordinates