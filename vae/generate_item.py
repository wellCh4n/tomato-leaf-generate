import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from dataset.tomato_dataset import TomatoLeafDataset

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='/home/featurize/tomato-leaf-generate/vae/models/vae_checkpoint_epoch_490.pth', type=str, help="模型检查点路径")
parser.add_argument("--output_path", type=str, default="vae_images", help="输出图像的路径")
parser.add_argument("--n_samples", type=int, default=100, help="重建图像的数量")
parser.add_argument("--batch_size", type=int, default=10, help="每批次处理的图像数量")
parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
args = parser.parse_args()

# 确保输出目录存在
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_path)
os.makedirs(output_path, exist_ok=True)

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


# 数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集
dataset = TomatoLeafDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
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
print(f"开始重建 {args.n_samples} 张图像...")
_trans = ToPILImage()
count = 0

with torch.no_grad():
    for batch_idx, imgs in enumerate(dataloader):
        if count >= args.n_samples:
            break

        # 将图像移至设备
        imgs = imgs.to(device)

        # 通过VAE进行重建
        recon_imgs, _, _ = vae(imgs)

        # 保存每对原始图像和重建图像
        for i in range(imgs.size(0)):
            if count >= args.n_samples:
                break

            # 保存原始图像
            # orig_img = _trans((imgs[i].cpu() * 0.5 + 0.5).clamp(0, 1))
            # orig_img.save(os.path.join(output_path, f"{count}_original.jpg"))

            # 保存重建图像
            recon_img = _trans((recon_imgs[i].cpu() * 0.5 + 0.5).clamp(0, 1))
            recon_img.save(os.path.join(output_path, f"{count}_reconstructed.jpg"))

            count += 1
            print(f"已处理 {count}/{args.n_samples} 张图像")

print(f"重建完成! 图像已保存到 {output_path}")