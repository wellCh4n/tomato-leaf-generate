import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import ToPILImage

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='/home/featurize/tomato-leaf-generate/gan/models2/checkpoint_epoch_490.pth', type=str, help="生成器模型检查点路径")
parser.add_argument("--output_path", type=str, default="gan_images", help="输出图像的路径")
parser.add_argument("--n_samples", type=int, default=100, help="生成图像的数量")
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


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4  # 初始特征图大小
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# 初始化生成器
generator = Generator().to(device)

# 加载模型
checkpoint = torch.load(args.model_path, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
print(f"生成器模型已从 {args.model_path} 加载")

# 设置为评估模式
generator.eval()

# 生成图像
print(f"开始生成 {args.n_samples} 张图像...")
_trans = ToPILImage()
with torch.no_grad():
    for i in tqdm(range(args.n_samples)):
        # 从正态分布采样潜在向量
        z = torch.randn(1, args.latent_dim).to(device)

        # 通过生成器生成图像
        generated_img = generator(z)

        # 将图像转换为PIL图像并保存
        img = _trans((generated_img[0].cpu() * 0.5 + 0.5).clamp(0, 1))
        img.save(os.path.join(output_path, f"{i}.jpg"))

print(f"生成完成! 图像已保存到 {output_path}")