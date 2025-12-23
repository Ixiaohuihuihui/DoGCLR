import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchlight import import_class
from scipy.spatial.transform import Rotation as R


class RotationAugmentation:
    """多视角旋转数据增强"""

    def __init__(self, small_angle=15, large_angle=60):
        self.small_angle = small_angle  # 普通旋转角度范围 (±15°)
        self.large_angle = large_angle  # 极端旋转角度范围 (±60°)

    def __call__(self, skeleton_data):
        """
        对骨架数据应用旋转增强
        输入: [B, C, T, V] 其中C=3 (x,y,z坐标)
        返回: (普通旋转, 极端旋转)
        """
        batch_size, channels, timesteps, joints = skeleton_data.shape
        assert channels == 3, "输入数据应为3D坐标"

        # 普通旋转
        small_rotated = self.apply_rotation(skeleton_data, self.small_angle)
        # 极端旋转
        large_rotated = self.apply_rotation(skeleton_data, self.large_angle)

        return small_rotated, large_rotated

    def apply_rotation(self, data, max_angle):
        """应用随机旋转到骨架数据"""
        rotated_data = torch.zeros_like(data)

        for i in range(data.size(0)):
            # 生成随机旋转角度
            angles = np.random.uniform(-max_angle, max_angle, size=3)

            # 创建旋转矩阵
            rot_matrix = self.generate_rotation_matrix(angles)

            # 应用旋转到所有时间步和关节
            for t in range(data.size(2)):
                for j in range(data.size(3)):
                    point = data[i, :, t, j]
                    rotated_point = torch.matmul(rot_matrix, point)
                    rotated_data[i, :, t, j] = rotated_point

        return rotated_data

    def generate_rotation_matrix(self, angles_deg):
        """生成绕x,y,z轴的旋转矩阵"""
        # 转换为弧度
        angles_rad = np.radians(angles_deg)

        # 创建绕x轴的旋转
        rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
            [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]
        ], dtype=torch.float32)

        # 创建绕y轴的旋转
        ry = torch.tensor([
            [np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
            [0, 1, 0],
            [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]
        ], dtype=torch.float32)

        # 创建绕z轴的旋转
        rz = torch.tensor([
            [np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
            [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 组合旋转矩阵: R = Rz * Ry * Rx
        rotation_matrix = torch.matmul(rz, torch.matmul(ry, rx))
        return rotation_matrix


class MeanPoseCalculator:
    """计算骨架序列的平均姿态"""

    def __init__(self):
        pass

    def __call__(self, skeleton_data):
        """
        输入: [B, C, T, V] 骨架序列
        返回: [B, C, 1, V] 平均姿态
        """
        # 在时间维度上取平均 (dim=2)
        mean_pose = skeleton_data.mean(dim=2, keepdim=True)
        return mean_pose


class RMCL(nn.Module):
    """基于多视角极小极大博弈的自监督骨架行为识别模型"""

    def __init__(self, base_encoder=None, feature_dim=128, queue_size=32768,
                 momentum=0.999, temperature=0.07, lambda_mi=0.5,
                 in_channels=3, hidden_channels=64, hidden_dim=256,
                 num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        # 加载基础编码器 (如ST-GCN)
        base_encoder = import_class(base_encoder)

        # 初始化查询编码器
        self.encoder_q = base_encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            num_class=feature_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
            **kwargs
        )

        # 初始化关键编码器 (动量更新)
        self.encoder_k = base_encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            num_class=feature_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
            **kwargs
        )

        # 投影头 (MLP)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_q.fc
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            self.encoder_k.fc
        )

        # 初始化关键编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 队列参数
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.lambda_mi = lambda_mi  # 互信息损失权重

        # 创建队列
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 平均姿态计算器
        self.mean_pose_calculator = MeanPoseCalculator()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新关键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 队列已满时替换最旧样本
        if ptr + batch_size > self.queue_size:
            # 计算剩余空间
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q_small, im_q_large, im_k):
        """
        前向传播
        输入:
            im_q_small: 普通旋转增强的查询序列 [B, C, T, V]
            im_q_large: 极端旋转增强的查询序列 [B, C, T, V]
            im_k: 关键序列 [B, C, T, V]
        """
        # 计算平均姿态
        mean_pose = self.mean_pose_calculator(im_q_small)  # [B, C, 1, V]

        # 提取特征
        q_small = self.encoder_q(im_q_small)  # 普通旋转特征
        q_large = self.encoder_q(im_q_large)  # 极端旋转特征
        q_mean = self.encoder_q(mean_pose)  # 平均姿态特征

        # 归一化特征
        q_small = F.normalize(q_small, dim=1)
        q_large = F.normalize(q_large, dim=1)
        q_mean = F.normalize(q_mean, dim=1)

        # 更新关键编码器并提取关键特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # 计算普通旋转视角的对比损失
        loss_small, logits_small = self.compute_contrastive_loss(q_small, q_mean, k)

        # 计算极端旋转视角的对比损失
        loss_large, logits_large = self.compute_contrastive_loss(q_large, q_mean, k)

        # 计算互信息损失 (最小化视角间冗余)
        mi_loss = self.compute_mutual_information_loss(logits_small, logits_large)

        # 更新队列
        self._dequeue_and_enqueue(k)

        # 总损失 = (普通损失 + 极端损失)/2 + λ * 互信息损失
        total_loss = (loss_small + loss_large) / 2 + self.lambda_mi * mi_loss

        return {
            'loss': total_loss,
            'loss_small': loss_small,
            'loss_large': loss_large,
            'mi_loss': mi_loss
        }

    def compute_contrastive_loss(self, q, anchor, k):
        """
        计算单个视角的对比损失
        q: 查询特征 (旋转增强)
        anchor: 锚点特征 (平均姿态)
        k: 关键特征
        """
        # 正样本相似度: q与锚点
        l_pos = torch.einsum('nc,nc->n', [q, anchor]).unsqueeze(-1)  # [B, 1]

        # 负样本相似度: q与队列中的样本
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B, K]

        # 组合logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, K+1]
        logits /= self.temperature

        # 标签: 第一个位置是正样本
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss, logits

    def compute_mutual_information_loss(self, logits_small, logits_large):
        """
        计算视角间的互信息损失 (KL散度)
        目标: 最小化两个视角特征分布之间的互信息
        """
        # 计算两个视角的条件概率分布
        p_small = F.softmax(logits_small, dim=1)
        p_large = F.softmax(logits_large, dim=1)

        # 计算KL散度 (对称形式)
        kl_small_large = F.kl_div(
            p_small.log(), p_large, reduction='batchmean',
            log_target=False
        )

        kl_large_small = F.kl_div(
            p_large.log(), p_small, reduction='batchmean',
            log_target=False
        )

        # 使用对称KL散度作为互信息损失
        mi_loss = (kl_small_large + kl_large_small) / 2

        return mi_loss


class RMCLTrainer:
    """RMCL模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # 初始化模型
        self.model = RMCL(
            base_encoder=config['base_encoder'],
            feature_dim=config['feature_dim'],
            queue_size=config['queue_size'],
            momentum=config['momentum'],
            temperature=config['temperature'],
            lambda_mi=config['lambda_mi'],
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            hidden_dim=config['hidden_dim'],
            num_class=config['num_class'],
            dropout=config['dropout'],
            graph_args=config['graph_args']
        ).to(self.device)

        # 初始化数据增强
        self.rotation_aug = RotationAugmentation(
            small_angle=config['small_angle'],
            large_angle=config['large_angle']
        )

        # 优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config['lr'],
            momentum=config['sgd_momentum'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma']
        )

        # 损失记录
        self.loss_history = {
            'total': [],
            'small': [],
            'large': [],
            'mi': []
        }

    def train_epoch(self, data_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, skeleton_data in enumerate(data_loader):
            # 移动到设备
            skeleton_data = skeleton_data.to(self.device)

            # 应用多视角旋转增强
            im_q_small, im_q_large = self.rotation_aug(skeleton_data)
            im_k = skeleton_data  # 原始数据作为关键

            # 前向传播
            outputs = self.model(im_q_small, im_q_large, im_k)

            # 计算损失
            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            self.loss_history['total'].append(loss.item())
            self.loss_history['small'].append(outputs['loss_small'].item())
            self.loss_history['large'].append(outputs['loss_large'].item())
            self.loss_history['mi'].append(outputs['mi_loss'].item())

            # 打印进度
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(data_loader)} | '
                      f'Loss: {loss.item():.4f} | Loss_small: {outputs["loss_small"].item():.4f} | '
                      f'Loss_large: {outputs["loss_large"].item():.4f} | MI Loss: {outputs["mi_loss"].item():.4f}')

        # 更新学习率
        self.scheduler.step()

        return total_loss / len(data_loader)

    def evaluate(self, data_loader):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for skeleton_data in data_loader:
                skeleton_data = skeleton_data.to(self.device)

                # 应用多视角旋转增强
                im_q_small, im_q_large = self.rotation_aug(skeleton_data)
                im_k = skeleton_data

                # 前向传播
                outputs = self.model(im_q_small, im_q_large, im_k)

                # 累加损失
                total_loss += outputs['loss'].item()

        return total_loss / len(data_loader)

    def save_checkpoint(self, epoch, path):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
        }, path)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_history = checkpoint['loss_history']
        return checkpoint['epoch']


