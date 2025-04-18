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
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
opt = parser.parse_args()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VAE模型定义
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(opt.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 潜在空间均值和对数方差
        self.fc_mu = nn.Linear(256 * (opt.img_size // 16) ** 2, opt.latent_dim)
        self.fc_var = nn.Linear(256 * (opt.img_size // 16) ** 2, opt.latent_dim)

        # 解码器
        self.decoder_input = nn.Linear(opt.latent_dim, 256 * (opt.img_size // 16) ** 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, opt.channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        x = x.view(-1, 256, opt.img_size // 16, opt.img_size // 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# 损失函数
def loss_function(recon_x, x, mu, log_var, kld_weight=0.00025):
    # MSE损失 (batch平均)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL散度 (batch平均)
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # 组合损失
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss


if __name__ == '__main__':
    # 初始化模型
    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 数据集和数据加载器
    dataset = TomatoLeafDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # 创建目录
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # TensorBoard日志
    writer = SummaryWriter(os.path.join("runs", "VAE"))

    # 训练循环
    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)

            optimizer.zero_grad()
            recon_imgs, mu, log_var = vae(imgs)
            loss, recon_loss, kld_loss = loss_function(recon_imgs, imgs, mu, log_var)

            loss.backward()
            optimizer.step()

            # 记录到TensorBoard
            writer.add_scalar("Total Loss", loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar("Reconstruction Loss", recon_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar("KL Divergence Loss", kld_loss.item(), epoch * len(dataloader) + i)

            # 打印训练进度
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Recon: %f] [KLD: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), recon_loss.item(), kld_loss.item())
            )

            # 保存样本图像
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(recon_imgs.data[:8], "images/%d.png" % batches_done, nrow=8, normalize=True)

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"models/vae_checkpoint_epoch_{epoch}.pth")
            print(f"Epoch {epoch} 检查点已保存!")

    writer.close()