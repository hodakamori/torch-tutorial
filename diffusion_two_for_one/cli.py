from utils import smiles2structure, get_gradients, add_diffusion_noise
from model import Net

smiles = "CCO"
coordinates, atomic_symbols, bonds = smiles2structure(smiles)

#Input
#coordinate, atomic_symbols, bond

model = Net(num_atoms=len(coordinates), num_node_features=64)

# Get force without noise
energy = model(coordinates, bonds, noise_level=2)
noise_pred = get_gradients(coordinates, energy)

# Get force with noise
noised_coordinates = add_diffusion_noise(coordinates, noise_level=2)
noise_true = noised_coordinates - coordinates

print(noise_pred)
# Get difference between with or without noise
loss = noise_true - noise_pred
loss.sum().backward()