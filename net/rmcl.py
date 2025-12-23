import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from aimclr import AimCLR


class RMCL(AimCLR):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3,
                 hidden_channels=64, hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__(base_encoder, pretrain, feature_dim, queue_size, momentum,
                         Temperature, mlp, in_channels, hidden_channels, hidden_dim,
                         num_class, dropout, graph_args, edge_importance_weighting, **kwargs)

        # 添加平均姿态计算模块
        self.register_buffer('mean_pose', torch.zeros(1, in_channels, 25, 2))  # 示例维度

        # 互信息优化参数
        self.lambda_mi = 0.5  # 互信息损失权重

    def compute_mean_pose(self, x):
        """计算输入骨架序列的全局平均姿态"""
        return x.mean(dim=1, keepdim=True)  # 时间维度平均

    def forward(self, im_q_small, im_q_large, im_k=None):
        """
        输入：
            im_q_small: 普通（小角度）增强的查询序列 [B, C, T, V]
            im_q_large: 极端（大角度）增强的查询序列
            im_k: 动量编码器的键序列
        """
        # 计算平均姿态
        x_mean = self.compute_mean_pose(im_q_small)  # [B, 1, C, V]

        # 多视角特征提取
        q_small = self.encoder_q(im_q_small)  # 普通旋转特征
        q_large = self.encoder_q(im_q_large)  # 极端旋转特征
        q_mean = self.encoder_q(x_mean)  # 平均姿态特征

        # 归一化
        q_small = F.normalize(q_small, dim=1)
        q_large = F.normalize(q_large, dim=1)
        q_mean = F.normalize(q_mean, dim=1)

        # 动量更新键编码器
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # 对比损失计算
        ## 普通旋转 vs 平均姿态
        l_pos_small = torch.einsum('nc,nc->n', [q_small, q_mean]).unsqueeze(-1)
        l_neg_small = torch.einsum('nc,ck->nk', [q_small, self.queue.clone().detach()])
        logits_small = torch.cat([l_pos_small, l_neg_small], dim=1) / self.T

        ## 极端旋转 vs 平均姿态
        l_pos_large = torch.einsum('nc,nc->n', [q_large, q_mean]).unsqueeze(-1)
        l_neg_large = torch.einsum('nc,ck->nk', [q_large, self.queue.clone().detach()])
        logits_large = torch.cat([l_pos_large, l_neg_large], dim=1) / self.T

        # 互信息计算（KL散度）
        p_small = F.softmax(logits_small, dim=1)
        p_large = F.softmax(logits_large, dim=1)
        mi_loss = F.kl_div(p_small.log(), p_large, reduction='batchmean')

        # 总损失 = 对比损失 + λ * 互信息损失
        loss_small = F.cross_entropy(logits_small, torch.zeros(logits_small.size(0)).long().cuda())
        loss_large = F.cross_entropy(logits_large, torch.zeros(logits_large.size(0)).long().cuda())
        total_loss = (loss_small + loss_large) / 2 + self.lambda_mi * mi_loss

        # 更新队列
        self._dequeue_and_enqueue(k)

        return {
            'loss': total_loss,
            'logits_small': logits_small,
            'logits_large': logits_large,
            'mi_loss': mi_loss
        }

    def _momentum_update_key_encoder(self):
        """动量更新键编码器（继承自AimCLR）"""
        super()._momentum_update_key_encoder()
