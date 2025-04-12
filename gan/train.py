import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import load_dataset

from dataset.tomato_dataset import TomatoLeafDataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr1", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4  # 初始特征图大小
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # 计算卷积后的特征图大小
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# 添加数据预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

tomato_dataset = TomatoLeafDataset(transform=transform)

dataloader = torch.utils.data.DataLoader(tomato_dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr2, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 在训练循环前添加
os.makedirs("models", exist_ok=True)
best_g_loss = float('inf')

logger = SummaryWriter(os.path.join("runs", "GAN"))
l = len(dataloader)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        batch_size = imgs.size(0)

        # 使用标签平滑化
        valid_smooth = Variable(Tensor(batch_size, 1).fill_(0.9), requires_grad=False)  # 原来是1.0
        fake_smooth = Variable(Tensor(batch_size, 1).fill_(0.1), requires_grad=False)  # 原来是0.0

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # 添加噪声到真实图像
        real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # ---------------------
        #  训练判别器 (多次训练判别器)
        # ---------------------

        # 每训练5次判别器，训练1次生成器
        d_iters = 1 if i % 1 == 0 else 1

        for _ in range(d_iters):
            optimizer_D.zero_grad()

            # 测量判别器区分真实样本和生成样本的能力
            real_loss = adversarial_loss(discriminator(real_imgs_noisy), valid_smooth)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_smooth)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 如果判别器太强，提前停止训练
            if d_loss.item() < 0.1:
                break

        # -----------------
        #  训练生成器 (仅当判别器表现不是太好时)
        # -----------------

        # 只有当判别器损失大于阈值时才训练生成器
        if d_loss.item() > 0.2:
            optimizer_G.zero_grad()

            # 生成器的损失衡量其欺骗判别器的能力
            g_loss = adversarial_loss(discriminator(gen_imgs), valid_smooth)

            g_loss.backward()
            optimizer_G.step()

        # 打印训练进度
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        logger.add_scalar("D_loss", d_loss.item(), global_step=epoch * l + i)
        logger.add_scalar("G_loss", g_loss.item(), global_step=epoch * l + i)

        if i % 100 == 0:
            logger.add_images("generated_images", (gen_imgs.data[:8] * 0.5 + 0.5), global_step=epoch * l + i)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:8], "images/%d.png" % batches_done, nrow=8, normalize=True)

    # 每10个epoch保存一次检查点
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'best_g_loss': best_g_loss
        }, f"models/checkpoint_epoch_{epoch}.pth")
        print(f"Epoch {epoch} 检查点已保存!")
