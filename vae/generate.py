import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torchvision.utils import save_image

from dataset.tomato_dataset import TomatoLeafDataset

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
parser.add_argument("--output_path", type=str, default="generated", help="输出图像的路径")
parser.add_argument("--n_samples", type=int, default=16, help="生成图像的数量")
parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_path, exist_ok=True)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# VAE模型定义
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(args.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 潜在空间均值和对数方差
        self.fc_mu = nn.Linear(256 * (args.img_size // 16) ** 2, args.latent_dim)
        self.fc_var = nn.Linear(256 * (args.img_size // 16) ** 2, args.latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(args.latent_dim, 256 * (args.img_size // 16) ** 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, args.channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, args.img_size // 16, args.img_size // 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = TomatoLeafDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
print(f"数据集加载完成，共 {len(dataset)} 张图像")

# 初始化模型
vae = VAE().to(device)

# 加载模型
checkpoint = torch.load(args.model_path, map_location=device)
vae.load_state_dict(checkpoint['model_state_dict'])
print(f"模型已从 {args.model_path} 加载")

# 设置为评估模式
vae.eval()

# 重建图像
with torch.no_grad():
    for i, imgs in enumerate(dataloader):
        # 将图像移至设备
        imgs = imgs.to(device)

        # 通过VAE进行重建
        recon_imgs, _, _ = vae(imgs)

        # 保存原始图像和重建图像
        # comparison = torch.cat([imgs, recon_imgs])
        output_file = os.path.join(args.output_path, f"reconstruction_batch_{i}.png")
        save_image(recon_imgs, output_file, nrow=8, normalize=True)
        print(f"批次 {i} 的重建图像已保存到 {output_file}")

        # 只处理第一个批次
        if i == 0:
            break

print("重建完成!")