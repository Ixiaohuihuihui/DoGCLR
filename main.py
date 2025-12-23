# torchlight
import torchlight as torchlight
from torchlight import import_class

from processor.processor import init_seed
from processor.pretrain_rmcl import RMCLTrainer
from dataloader.ntu_loader import create_data_loader
init_seed(0)

# 配置示例
config = {
    'base_encoder': 'net.st_gcn.Model',  # ST-GCN模型路径
    'feature_dim': 128,
    'queue_size': 32768,
    'momentum': 0.999,
    'temperature': 0.07,
    'lambda_mi': 0.5,
    'in_channels': 3,  # x, y, z坐标
    'hidden_channels': 64,
    'hidden_dim': 256,
    'num_class': 60,  # NTU-RGB+D数据集类别数
    'dropout': 0.5,
    'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
    'small_angle': 15,  # 普通旋转角度范围
    'large_angle': 60,  # 极端旋转角度范围
    'device': 'cuda:0',
    'lr': 0.1,
    'sgd_momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_milestones': [250],  # 在第250个epoch降低学习率
    'lr_gamma': 0.1,
    'log_interval': 50,  # 每50个batch打印一次日志
    'num_epochs': 3    # default: 300
}

def main():
    # 初始化训练器
    trainer = RMCLTrainer(config)

    # 加载数据集
    train_loader = create_data_loader('train')
    val_loader = create_data_loader('val')

    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(1, config['num_epochs'] + 1):
        trainer.current_epoch = epoch

        # 训练一个epoch
        train_loss = trainer.train_epoch(train_loader)
        print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')

        # 验证
        val_loss = trainer.evaluate(val_loader)
        print(f'Epoch {epoch} | Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, f'best_model_epoch_{epoch}.pth')
            print(f'Saved best model at epoch {epoch}')

        # 定期保存检查点
        if epoch % 50 == 0:
            trainer.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')


if __name__ == '__main__':
    main()

