"""Quick profiling script to estimate model size (Params, MB), compute (GFLOPs / GMACs),
and inference latency (ms/video, FPS) for the AEM model on CaptainCook4D / EgoPER.

Breaks down metrics by sub-module:
  1. Action Segmentation branch (backbone + neck + cls_head + reg_head)
  2. Effect Modeling branch (effect_model)
  3. Mistake Detection branch (action_model) + optional frozen OpenCLIP
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UTILS_DIR = os.path.join(REPO_ROOT, "libs", "utils")
for _p in (REPO_ROOT, UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from libs.core import load_config
from libs.modeling import make_meta_arch

warnings.filterwarnings("ignore")


def count_params(module: nn.Module):
    """Return (total_params, trainable_params, size_mb_fp32)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in module.buffers())
    size_mb = (total + buffers) * 4 / (1024 ** 2)
    return total, trainable, size_mb


def make_dummy_video_list(seq_len: int, feat_dim: int = 2048, fps: float = 10.0, num_classes: int = 13):
    """Create a 1-video dummy input matching CaptainCook4D inference format."""
    # Create ~8 synthetic ground-truth segments so postprocessing has reference labels
    seg_bounds = np.linspace(0, seq_len - 1, 9, dtype=int)
    segs = [[int(seg_bounds[i]), int(seg_bounds[i + 1] - 1)] for i in range(8)]
    labels = [int(i % num_classes) for i in range(8)]
    return [
        {
            "video_id": "dummy_video",
            "feats": torch.randn(feat_dim, seq_len),
            "segments": torch.tensor(segs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "labels_error": torch.zeros(len(segs), dtype=torch.long),
            "fps": fps,
            "duration": seq_len / fps,
            "visual_info": None,
            "scene_graph": None,
        }
    ]


@torch.no_grad()
def run_seg_step(model, video_list):
    fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = model.extract_feature(video_list)
    results = model.segment_video(video_list, points, fpn_masks, cls_prob_logits, out_offsets)
    return fpn_feats, fpn_masks, results


@torch.no_grad()
def run_effect_step(model, fpn_feats, fpn_masks, results, num_classes):
    vid_result = list(results.values())[0]
    segs = vid_result["segments"]
    pred_labels = vid_result["segment_action_pred"]
    # With random untrained weights, predicted labels might be all 0 (BG);
    # ensure at least a few non-BG segments so effect_model.inference runs its transformer
    if (pred_labels > 0).sum() == 0 and len(pred_labels) > 0:
        pred_labels = pred_labels.clone()
        pred_labels[: min(5, len(pred_labels))] = 1
    segment_feature = model.generate_segment_features(fpn_feats, fpn_masks, [segs])
    enhanced = model.effect_model.inference(
        feature_segments=segment_feature,
        action_labels=[pred_labels],
        device=model.device,
    )
    return enhanced, pred_labels


def main():
    parser = argparse.ArgumentParser(description="Profile model size, GFLOPs, and latency.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"),
        help="path to config file (default: configs/ccp4d.yaml)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="optional checkpoint file (.pth.tar) to measure actual disk size and load weights",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2304,
        help="input sequence length in frames @ 10fps (default: 2304 = 230.4s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="number of timing runs after warmup (default: 10)",
    )
    parser.add_argument("--use_gcn", action="store_true", help="use GCN backbone (EgoPER)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg["model"]["backbone_type"] = "convGCNTransformer" if args.use_gcn else "convTransformer"
    model = make_meta_arch(cfg["model_name"], **cfg["model"]).to(device).eval()

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_mb = os.path.getsize(args.ckpt) / (1024 ** 2)
        ckpt_data = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Adjust num_classes if checkpoint differs
        if "cls_head.cls_head.conv.bias" in clean_sd:
            ckpt_classes = clean_sd["cls_head.cls_head.conv.bias"].shape[0]
            if ckpt_classes != cfg["model"]["num_classes"]:
                cfg["model"]["num_classes"] = ckpt_classes
                model = make_meta_arch(cfg["model_name"], **cfg["model"]).to(device).eval()
        model.load_state_dict(clean_sd, strict=False)
    else:
        ckpt_mb = None

    # 1. Parameter count & memory breakdown
    seg_modules = nn.ModuleList([model.backbone, model.neck, model.cls_head, model.reg_head])
    tot_p, train_p, tot_mb = count_params(model)
    seg_p, _, seg_mb = count_params(seg_modules)
    eff_p, _, eff_mb = count_params(model.effect_model)
    act_p, _, act_mb = count_params(model.action_model)

    print("=" * 80)
    print(f" 1. KÍCH THƯỚC MÔ HÌNH (MODEL SIZE & PARAMETERS) — Device: {device}")
    print("=" * 80)
    print(f"{'Thành phần (Sub-module)':<42} {'Params (M)':>12} {'Tỷ lệ (%)':>11} {'RAM FP32 (MB)':>14}")
    print("-" * 80)
    print(f"{'Nhánh Segmentation (Backbone+Neck+Heads)':<42} {seg_p/1e6:12.3f} {seg_p/tot_p*100:10.1f}% {seg_mb:14.2f}")
    print(f"{'  ├─ Backbone (' + cfg['model']['backbone_type'] + ')':<42} {count_params(model.backbone)[0]/1e6:12.3f}")
    print(f"{'  ├─ClsHead (DynClsHead)':<42} {count_params(model.cls_head)[0]/1e6:12.3f}")
    print(f"{'  └─RegHead (DynRegHead)':<42} {count_params(model.reg_head)[0]/1e6:12.3f}")
    print(f"{'Nhánh Effect Modeling (Effect_Model)':<42} {eff_p/1e6:12.3f} {eff_p/tot_p*100:10.1f}% {eff_mb:14.2f}")
    print(f"{'Nhánh Mistake Detector (Action_Model)':<42} {act_p/1e6:12.3f} {act_p/tot_p*100:10.1f}% {act_mb:14.2f}")
    print("-" * 80)
    print(f"{'TỔNG CỘNG (AEM PtTransformer)':<42} {tot_p/1e6:12.3f} {'100.0%':>11} {tot_mb:14.2f}")
    if ckpt_mb is not None:
        print(f"{'Dung lượng file .pth.tar trên ổ đĩa':<42} {'(checkpoint + EMA)':>24} {ckpt_mb:14.2f}")
    print(f"{'[Lưu ý] OpenCLIP EVA02-L-14-336 (Frozen)':<42} {'~428.000':>12} {'(ngoài ckpt)':>11} {'~1712.00':>14}")
    print("=" * 80)

    # 2. Compute (FLOPs / MACs)
    num_classes = cfg["model"]["num_classes"]
    video_list = make_dummy_video_list(
        seq_len=args.seq_len,
        feat_dim=cfg["model"]["input_dim"],
        fps=10.0,
        num_classes=num_classes,
    )

    with FlopCounterMode(display=False) as flop_counter_seg:
        fpn_feats, fpn_masks, results = run_seg_step(model, video_list)
    seg_flops = flop_counter_seg.get_total_flops()

    with FlopCounterMode(display=False) as flop_counter_eff:
        enhanced, pred_labels = run_effect_step(model, fpn_feats, fpn_masks, results, num_classes)
    eff_flops = flop_counter_eff.get_total_flops()

    tot_flops = seg_flops + eff_flops
    vid_duration_sec = args.seq_len / 10.0

    print(f"\n 2. SỐ LƯỢNG TÍNH TOÁN (FLOPs / MACs) — Với 1 video dài {args.seq_len} frames ({vid_duration_sec:.1f}s @ 10fps)")
    print("=" * 80)
    print(f"{'Giai đoạn (Stage)':<45} {'GFLOPs':>15} {'GMACs (~FLOPs/2)':>18}")
    print("-" * 80)
    print(f"{'Action Segmentation (extract_feature + heads)':<45} {seg_flops/1e9:15.3f} {seg_flops/2e9:18.3f}")
    print(f"{'Effect Modeling (effect_transformer + MLP)':<45} {eff_flops/1e9:15.3f} {eff_flops/2e9:18.3f}")
    print(f"{'TỔNG (AEM không tính OpenCLIP text)':<45} {tot_flops/1e9:15.3f} {tot_flops/2e9:18.3f}")
    print("=" * 80)

    # 3. Latency / Speed
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        f_feats, f_masks, res = run_seg_step(model, video_list)
        _ = run_effect_step(model, f_feats, f_masks, res, num_classes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    seg_times = []
    eff_times = []
    for _ in range(args.runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        f_feats, f_masks, res = run_seg_step(model, video_list)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = run_effect_step(model, f_feats, f_masks, res, num_classes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        seg_times.append((t1 - t0) * 1000.0)
        eff_times.append((t2 - t1) * 1000.0)

    mean_seg_ms = float(np.mean(seg_times))
    mean_eff_ms = float(np.mean(eff_times))
    mean_tot_ms = mean_seg_ms + mean_eff_ms
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )

    print(f"\n 3. THỜI GIAN CHẠY (INFERENCE LATENCY) — Trung bình qua {args.runs} lần chạy")
    print("=" * 80)
    print(f"  - Thời gian chạy nhánh Segmentation ( + NMS) : {mean_seg_ms:7.2f} ms / video")
    print(f"  - Thời gian chạy nhánh Effect Enricher       : {mean_eff_ms:7.2f} ms / video")
    print(f"  - TỔNG thời gian suy luận 1 video ({vid_duration_sec:.0f}s)     : {mean_tot_ms:7.2f} ms / video (~{1000.0/mean_tot_ms:.1f} videos/giây)")
    print(f"  - Tốc độ xử lý frame (Throughput)            : {args.seq_len / (mean_tot_ms / 1000.0):7.0f} frames/giây (x{vid_duration_sec / (mean_tot_ms / 1000.0):.0f} lần thời gian thực)")
    if torch.cuda.is_available():
        print(f"  - Đỉnh bộ nhớ GPU (Peak VRAM allocated)      : {peak_vram_mb:7.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()

