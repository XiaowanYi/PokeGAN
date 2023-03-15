import torch
from aegan import Generator as G
import torchvision.utils as vutils
import os
from tqdm import tqdm

num_of_pokemons = 20
out_dir = 'NewPokemons'
device = torch.device('cpu')
netG = G()
netG.load_state_dict(torch.load('trained_generator_weights.pt', map_location=device))
vec = torch.randn((32, 16))
with torch.no_grad():
    fake = netG(vec)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for i in tqdm(range(num_of_pokemons)):
    vutils.save_image(fake.data[i], f'{out_dir}/creature.{i:02d}.png', normalize=True)
