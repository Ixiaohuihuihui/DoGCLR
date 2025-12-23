import torch

# 加载保存的模型或检查点
ckpt_path = '/data/home/mk/RMCL/checkpoints/model_epoch_20.pth'  # 替换成你的路径
checkpoint = torch.load(ckpt_path, map_location='cpu')  # 用 CPU 加载，避免显存不够

# 打印这个文件的顶层类型和键
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())

    # 如果是 state_dict 模式（常见），可能是直接参数字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # 有些直接就是 state_dict

    # 打印所有参数名称及形状
    for name, param in state_dict.items():
        print(name, param.shape if hasattr(param, 'shape') else type(param))
else:
    print("Unexpected checkpoint format:", checkpoint)
