import torch
import matplotlib.pyplot as plt
import os

# ==== 修改这里：加载你保存的 best_model.pth 或某个 epoch 的 checkpoint ====
ckpt_path = "/data/home/mk/RMCL/checkpoints/model_epoch_300.pth"
save_fig  = "loss_curve.png"

assert os.path.exists(ckpt_path), f"找不到 {ckpt_path}"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# ==== 从 checkpoint 里取出 loss_history ====
if "loss_history" in checkpoint:
    loss_hist = checkpoint["loss_history"]
    total_loss = loss_hist.get("total", [])
    small_loss = loss_hist.get("small", [])
    large_loss = loss_hist.get("large", [])
    mi_loss    = loss_hist.get("mi", [])
else:
    raise KeyError("这个 checkpoint 里没有 loss_history，可以换用保存时包含 loss_history 的文件")

# ==== 画图 ====
plt.figure(figsize=(8,6))
plt.plot(total_loss, label="Total loss")
plt.plot(small_loss, label="Small rot loss", alpha=0.7)
plt.plot(large_loss, label="Large rot loss", alpha=0.7)
plt.plot(mi_loss,    label="MI loss", alpha=0.7)

plt.xlabel("Iteration (batch)")
plt.ylabel("Loss")
plt.title("Pretrain Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_fig, dpi=150)
print(f"Saved figure -> {save_fig}")
