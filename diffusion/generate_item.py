import argparse
import os
import torch
from tqdm import tqdm
from torchvision.transforms import ToPILImage

from diffusion.modules import UNet
from diffusion.train import Diffusion

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/DDPM_Uncondtional/ckpt.pt",
                    help="模型检查点路径")
parser.add_argument("--output_path", type=str, default="diffusion_images", help="输出图像的路径")
parser.add_argument("--n_samples", type=int, default=100, help="生成图像的总数量")
parser.add_argument("--batch_size", type=int, default=10, help="每批次生成的图像数量")
parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
args = parser.parse_args()

# 确保输出目录存在
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_path)
os.makedirs(output_path, exist_ok=True)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型
model = UNet().to(device)

# 加载模型
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint)
print(f"模型已从 {args.model_path} 加载")

# 初始化Diffusion
diffusion = Diffusion(img_size=args.img_size, device=device)

# 计算需要生成的批次数
num_batches = args.n_samples // args.batch_size
if args.n_samples % args.batch_size != 0:
    num_batches += 1

# 生成图像
print(f"开始生成 {args.n_samples} 张图像，分 {num_batches} 批次进行...")
model.eval()

image_count = 0
with torch.no_grad():
    for batch in range(num_batches):
        # 计算当前批次应生成的图像数量
        current_batch_size = min(args.batch_size, args.n_samples - image_count)

        print(f"生成第 {batch + 1}/{num_batches} 批次，{current_batch_size} 张图像...")

        # 生成当前批次的图像
        x = diffusion.sample(model, n=current_batch_size)

        # 保存每张图像
        _trans = ToPILImage()
        for i, image in enumerate(tqdm(x)):
            img = _trans(image)
            img.save(os.path.join(output_path, f"{image_count + i}.jpg"))

        # 更新已生成的图像计数
        image_count += current_batch_size

        print(f"已完成 {image_count}/{args.n_samples} 张图像")

print(f"生成完成! 所有 {args.n_samples} 张图像已保存到 {output_path}")