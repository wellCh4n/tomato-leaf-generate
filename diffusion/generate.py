import torch

from diffusion.modules import UNet
from diffusion.train import Diffusion
from diffusion.utils import plot_images, save_images

device = "cuda"
model = UNet().to(device)
ckpt = torch.load("unconditional_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=16)
save_images(x, './generate.jpg')