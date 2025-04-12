import torch

from diffusion.modules import UNet
from diffusion.train import Diffusion
from diffusion.utils import plot_images, save_images, save_images_item

device = "cuda"
model = UNet().to(device)
ckpt = torch.load("/home/featurize/tomato-leaf-generate/models/DDPM_Uncondtional/ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=256, device=device)
x = diffusion.sample(model, n=64)
save_images_item(x, '.')