# from torchvision.datasets import ImageFolder
# import torch
# from torchvision import transforms
#
# def get_mean_and_std(train_data):
#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=1, shuffle=False, num_workers=0,
#         pin_memory=True)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for X, _ in train_loader:
#         for d in range(3):
#             mean[d] += X[:, d, :, :].mean()
#             std[d] += X[:, d, :, :].std()
#     mean.div_(len(train_data))
#     std.div_(len(train_data))
#     return list(mean.numpy()), list(std.numpy())
#
#
# if __name__ == '__main__':
#     train_dataset = ImageFolder(root=r'E:\code\rice_disease\demo\ConvNeXt\data\rice_diseases', transform=transforms.ToTensor())
#     print(get_mean_and_std(train_dataset))


from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from tqdm import tqdm  # 用于显示进度条


def get_mean_and_std(train_data, device='cuda'):
    # 将数据加载到GPU上计算
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,  # 适当增大batch_size以利用GPU并行计算
        shuffle=False,
        num_workers=1,  # 使用多线程加载数据
        pin_memory=True  # 启用锁页内存，加速数据传输到GPU
    )

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_pixels = 0

    with torch.no_grad():  # 不需要计算梯度
        for X, _ in tqdm(train_loader, desc='Calculating mean and std'):
            X = X.to(device, non_blocking=True)  # 将数据异步传输到GPU
            # 计算每个通道的均值和方差
            # [batch_size, channels, height, width] -> [channels]
            batch_pixels = X.size(0) * X.size(2) * X.size(3)
            mean += X.mean(dim=(0, 2, 3)) * batch_pixels
            std += X.std(dim=(0, 2, 3)) * batch_pixels
            total_pixels += batch_pixels

    mean /= total_pixels
    std /= total_pixels

    return mean.cpu().numpy().tolist(), std.cpu().numpy().tolist()


if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    ])

    # 加载数据集
    train_dataset = ImageFolder(
        root=r'E:\code\rice_disease\demo\ConvNeXt\data\rice_diseases',
        transform=transform
    )

    # 计算均值和标准差
    mean, std = get_mean_and_std(train_dataset, device=device)
    print(f"Mean: {mean}")
    print(f"Std: {std}")