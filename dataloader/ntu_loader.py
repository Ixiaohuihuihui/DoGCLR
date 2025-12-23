import os
from torch.utils.data import DataLoader
from feeder.NTUDatasets import NTUMotionProcessor, SimpleLoader

def create_data_loader(split: str):
    """
    根据 split ('train' / 'val') 创建并返回 DataLoader。
    优先使用 NTUMotionProcessor（标准 NTU 格式：.npy + label.pkl）。
    若检测不到 label.pkl，则回退到 SimpleLoader（position.npy / motion.npy / label.npy）。
    """
    assert split in ("train", "val")

    # ====== 1) 配置区：改成你自己的路径 ======
    # 标准 NTU-RGB+D（以 xsub 为例）
    ntu_root = r"D:\代码\RMCL\datasets\ntu60_frame50\xsub"
    protocol = "xsub"                              # xsub or xview
    data_file = os.path.join(ntu_root, f"{protocol}_{split}_data.npy")
    label_file_pkl = os.path.join(ntu_root, f"{protocol}_{split}_label.pkl")

    simple_data_file = os.path.join(ntu_root, f"{protocol}_{split}_data.npy")
    simple_label_npy = os.path.join(ntu_root, f"{protocol}_{split}_label.npy")


    # ====== 2) 训练用常见超参======
    batch_size = 64 if split == "train" else 64
    shuffle = (split == "train")
    num_workers = 0      # Windows 下建议先用 0，Linux 可开到 4/8
    pin_memory = True

    # ====== 3) 优先尝试 NTUMotionProcessor ======
    if os.path.exists(data_file) and os.path.exists(label_file_pkl):
        dataset = NTUMotionProcessor(
            data_path=data_file,
            label_path=label_file_pkl,
            mmap=True,
            data_type='normal',     # 或 'relative' / 'multi_relative'，看你实验需要
            t_length=300,           # 你上面类里 max_length=300，按需裁剪/重采样
            y_rotation=False,       # 需要做对齐再开
            displacement=False,     # >0 时会返回 (data, motion, label)
            sampling='force_crop'   # 或 'resize'
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    # ====== 4) 回退到 SimpleLoader（position/motion/label 都是 .npy）======
    if os.path.exists(simple_data_file) and os.path.exists(simple_label_npy):
        dataset = SimpleLoader(
            data_path=simple_data_file,   # 会自动推导 *_position.npy / *_motion.npy
            label_path=simple_label_npy,
            debug=False,
            mmap=True,
            data_type='relative',   # 或 'normal'
            displacement=0,         # >0 则返回 (position, motion, label)
            t_length=200,
            y_rotation=True,        # 如需平行化对齐
            sampling='resize'       # 或 'force_crop'
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    # ====== 5) 都找不到就清晰报错 ======
    raise FileNotFoundError(
        f"未找到数据文件：\n"
        f"  NTU格式期望: {data_file} + {label_file_pkl}\n"
        f"  SimpleLoader期望: {simple_data_file} + {simple_label_npy}\n"
        f"请检查路径或改成你本机的数据路径。"
    )