import torch
import random
from torch.utils.data import DataLoader, random_split

from dataset import CGCoordsDataset
from model import Net
from utils import smiles2structure, get_gradients, add_diffusion_noise

device = torch.device("cuda")
topology_path = "./ala2_cg.pdb"
traj_path = "./ala2_cg.xtc"
dataset = CGCoordsDataset(topology_path, traj_path)
print(len(dataset))
bonds = dataset.bonds

MAX_EPOCHS = 5
BATCH_SIZE = 64
max_noise_level = 10
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Net(num_atoms=5, num_node_features=64)
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, total_steps=100000
)
loss_func = torch.nn.MSELoss(reduction="none")

for epoch in range(MAX_EPOCHS):
    history = []
    for indices, coords in dataloader:
        coords.requires_grad_()
        noise_level = random.randint(1, max_noise_level + 1)
        noised_coordinates = add_diffusion_noise(coords, noise_level=noise_level)
        noise_true = noised_coordinates - coords
        energy = model(indices, coords, bonds, noise_level=noise_level)
        if coords.grad is not None:
            coords.grad.zero_()
        energy.backward(retain_graph=True)
        noise_pred = coords.grad
        loss = loss_func(noise_true, noise_pred).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        history.append(loss.detach().numpy())
    print(epoch, sum(history) / len(history))
