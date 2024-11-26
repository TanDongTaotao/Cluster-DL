import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from DCEC import DCEC

def main():
    parser = argparse.ArgumentParser(description='DCEC Training')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--update-interval', type=int, default=140)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='results/dcec')
    args = parser.parse_args()

    # 设置CUDA相关参数
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    
    # 使用固定的随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                 transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # 启用pin_memory加速数据传输
        num_workers=4     # 使用多个工作进程加载数据
    )

    # 初始化模型
    model = DCEC(input_shape=(1, 32, 32), n_clusters=args.n_clusters)
    
    # 训练模型
    model.fit(train_loader, 
             y=train_dataset.targets.numpy(),
             maxiter=args.epochs * len(train_loader),
             update_interval=args.update_interval,
             save_dir=args.save_dir)

if __name__ == '__main__':
    main() 