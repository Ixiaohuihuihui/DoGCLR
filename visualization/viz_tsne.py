#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, yaml, numpy as np, torch, torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchlight import import_class  # 你项目里已有

def build_model(model_name, model_args, device):
    Model = import_class(model_name)
    model = Model(**model_args)
    model.eval().to(device)
    return model

def load_weights(model, weights_path, strict=False):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(weights_path)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state: state = state["state_dict"]
        elif "model" in state:    state = state["model"]
    new_state = {k.replace("module.",""): v for k,v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing or unexpected:
        print(f"[INFO] missing: {missing}, unexpected: {unexpected}")

def replace_cls_with_identity(model):
    # 典型 AimCLR 线性层在 encoder_q.fc
    paths = ["encoder_q.fc", "encoder_q_bone.fc", "encoder_q_motion.fc", "fc"]
    for path in paths:
        obj, parent = model, model
        parts = path.split(".")
        ok = True
        for i,p in enumerate(parts):
            if hasattr(obj, p):
                parent, obj = obj, getattr(obj, p)
            else:
                ok = False; break
        if ok and isinstance(obj, nn.Module):
            setattr(parent, parts[-1], nn.Identity())
            print(f"[OK] 使用 '{path}' 的Identity输出特征")
            return
    print("[WARN] 没找到分类头，t-SNE将直接用当前forward的输出")

def make_loader(feeder_name, feeder_args, batch_size, num_workers):
    Feeder = import_class(feeder_name)
    ds = Feeder(**feeder_args)
    ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True, drop_last=False)
    return ld, ds

def collect_features(model, loader, device, stream="joint", limit=-1):
    feats, labels = [], []
    Bone = [(1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),
            (10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),
            (18,17),(19,18),(20,19),(21,21),(22,23),(23,8),(24,25),(25,12)]
    with torch.no_grad():
        for data, label in loader:
            data = data.float().to(device, non_blocking=True)
            if stream == "bone":
                bone = torch.zeros_like(data)
                for v1,v2 in Bone:
                    bone[:,:,:,v1-1,:] = data[:,:,:,v1-1,:] - data[:,:,:,v2-1,:]
                data = bone
            try:
                out = model(None, data)   # AimCLR常见签名
            except TypeError:
                out = model(data)
            feats.append(out.detach().cpu())
            labels.append(label)
            if limit > 0 and sum(x.shape[0] for x in feats) >= limit:
                break
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def main():
    ap = argparse.ArgumentParser("t-SNE for AimCLR features")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--output", default="tsne.png")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg["model"], cfg["model_args"], args.device)
    print(f"[INFO] loading weights: {args.weights}")
    load_weights(model, args.weights, strict=False)
    replace_cls_with_identity(model)

    if args.split == "test":
        feeder_name = cfg["test_feeder"]
        feeder_args = cfg["test_feeder_args"]
        bs = cfg.get("test_batch_size", 256)
    else:
        feeder_name = cfg["train_feeder"]
        feeder_args = cfg["train_feeder_args"]
        bs = cfg.get("batch_size", 256)
    num_workers = cfg.get("num_worker", 4)
    loader, _ = make_loader(feeder_name, feeder_args, bs, num_workers)

    stream = cfg.get("stream", "joint")
    print(f"[INFO] stream={stream}, split={args.split}, limit={args.limit}, perplexity={args.perplexity}")
    feats, labels = collect_features(model, loader, args.device, stream, args.limit)
    print(f"[INFO] feats: {feats.shape}, labels: {labels.shape}")

    from sklearn.manifold import TSNE
    print("[INFO] fitting TSNE...")
    emb = TSNE(n_components=2, init="pca", random_state=0, perplexity=args.perplexity).fit_transform(feats)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    sc = plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, alpha=0.75)
    plt.title("t-SNE of Features")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label("Class ID")
    plt.tight_layout(); plt.savefig(args.output, dpi=300)
    print(f"[OK] saved: {args.output}")

if __name__ == "__main__":
    main()
