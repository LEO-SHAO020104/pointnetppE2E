import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


# ------------------- 修改后的模型结构 -------------------
class PointNet2E2E(nn.Module):
    def __init__(self, num_points=3000):
        super(PointNet2E2E, self).__init__()
        # 输入通道调整为13（3坐标+10特征）
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32,
                                          in_channel=13, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # 调整特征传播通道匹配
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 13, mlp=[128, 64, 32])

        # 输出层调整为3个通道
        self.output_mlp = nn.Sequential(
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 3, 1)
        )
        self.num_points = num_points

    def forward(self, xyz_features):
        # 输入处理 [B, N, 13] -> [B, 13, N]
        xyz = xyz_features[:, :, :3].transpose(1, 2)
        features = xyz_features[:, :, 3:].transpose(1, 2)
        x = torch.cat([xyz, features], dim=1)

        # 编码器
        l1_xyz, l1_points = self.sa1(xyz, x)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # 解码器
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, torch.cat([xyz, features], 1), l1_points)

        # 输出预测
        return self.output_mlp(l0_points).transpose(1, 2)


# ------------------- 数据管道 -------------------
class PointCloudDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if split in f]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 加载npy文件，假设数据格式为：
        # 输入：[num_points, 13] (3坐标 + 10特征)
        # 标签：[num_points, 3] (3个预测值)
        data = np.load(self.data_files[idx])
        return {
            'input': data[:, :13].astype(np.float32),
            'label': data[:, 13:].astype(np.float32)
        }


# ------------------- 训练模块 -------------------
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化数据加载器
        train_dataset = PointCloudDataset(config['data_dir'], 'train')
        val_dataset = PointCloudDataset(config['data_dir'], 'val')

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        # 初始化模型
        self.model = PointNet2E2E().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            inputs = torch.tensor(batch['input']).to(self.device)
            labels = torch.tensor(batch['label']).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = torch.tensor(batch['input']).to(self.device)
                labels = torch.tensor(batch['label']).to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.config['model_dir']}/best_model.pth")

            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print('-' * 50)


# ------------------- 配置和运行 -------------------
if __name__ == "__main__":
    config = {
        'data_dir': './data',
        'model_dir': './checkpoints',
        'batch_size': 8,
        'epochs': 100,
        'lr': 0.001,
    }

    # 创建输出目录
    os.makedirs(config['model_dir'], exist_ok=True)

    # 启动训练
    trainer = Trainer(config)
    trainer.train()