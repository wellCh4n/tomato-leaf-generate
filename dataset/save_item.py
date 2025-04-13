import os
import argparse
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

from dataset.tomato_dataset import TomatoLeafDataset

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="real", help="输出图像的路径")
parser.add_argument("--n_samples", type=int, default=100, help="保存图像的数量")
parser.add_argument("--img_size", type=int, default=256, help="图像尺寸")
args = parser.parse_args()

# 确保输出目录存在
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_path)
os.makedirs(output_path, exist_ok=True)

# 加载数据集
dataset = TomatoLeafDataset(transform=None)  # 不应用变换，保持原始图像
print(f"数据集加载完成，共 {len(dataset)} 张图像")

# 创建图像转换器
_trans = ToPILImage()

# 保存图像
print(f"开始保存 {args.n_samples} 张真实图像...")
count = 0

for idx in tqdm(range(min(args.n_samples, len(dataset)))):
    # 获取图像
    img = dataset[idx]

    # 转换为PIL图像并保存
    pil_img = _trans(img)

    # 保存图像
    pil_img.save(os.path.join(output_path, f"{idx}.jpg"))
    count += 1

print(f"保存完成! 共保存了 {count} 张真实图像到 {output_path}")