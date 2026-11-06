#!/usr/bin/env python3
"""Timeline Comparison and Evaluation Script: Trained No-AEM vs. Paper Full-AEM.

Evaluates trained models against the author's paper checkpoints on CaptainCook4D
recipes and generates side-by-side timeline bar visualizations (PNG + ASCII)
without the overhead of decoding or rendering full-frame MP4 videos.

Outputs:
1. High-resolution matplotlib timeline PNG images showing stacked bars:
     - Ground Truth (GT)
     - Paper Full-AEM
     - Trained No-AEM
   with distinct colors, step names inside segments, and full legend.
2. Direct terminal ASCII timeline bars for instantaneous feedback in Colab stdout.
3. Side-by-side quantitative metrics table (Acc, Edit, F1, Frag, Prec, AUC, EDA).
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libs.core import load_config
from libs.datasets.clip_encode_ccp4d import action_tokenize_ccp4d
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from metric import calculate_micro_auc, evaluate_ed, framewise_to_segments

from test_segmentation import (
    DEFAULT_SPLIT_FILE,
    CombinedCaptainCook4DDataset,
    _resolve_root_dir,
    build_action_name_map,
    compute_ed_metrics,
    compute_extended_seg_metrics,
    load_clip_model,
    load_split_video_ids,
    load_weights,
    run_full_inference,
    run_segmentation_only,
)

from train_no_aem import configure_no_aem_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Color & Style Helpers
# -----------------------------------------------------------------------------

PRESET_COLORS = [
    "#E0E0E0",  # 0: Background (Light subtle gray)
    "#1F77B4",  # 1: Tab Blue
    "#FF7F0E",  # 2: Tab Orange
    "#2CA02C",  # 3: Tab Green
    "#D62728",  # 4: Tab Red
    "#9467BD",  # 5: Tab Purple
    "#8C564B",  # 6: Tab Brown
    "#E377C2",  # 7: Tab Pink
    "#17BECF",  # 8: Tab Cyan
    "#BCBD22",  # 9: Tab Olive
    "#3366CC",  # 10: Cobalt
    "#DC3912",  # 11: Vermilion
    "#FF9900",  # 12: Amber
    "#109618",  # 13: Emerald
    "#990099",  # 14: Violet
    "#0099C6",  # 15: Teal
    "#DD4477",  # 16: Rose
    "#66AA00",  # 17: Lime
    "#B82E2E",  # 18: Brick
    "#316395",  # 19: Steel
    "#994499",  # 20: Amethyst
    "#22AA99",  # 21: Aqua
    "#AAAA11",  # 22: Gold
    "#6633CC",  # 23: Indigo
    "#E67300",  # 24: Tangerine
    "#8B0707",  # 25: Maroon
    "#329262",  # 26: Forest
    "#5574A6",  # 27: Slate
    "#3B3EAC",  # 28: Deep Blue
]


def get_step_color(step_id: int):
    """Return a consistent HEX color for a given step class ID."""
    if step_id == 0:
        return "#E8ECEF"  # Background
    if step_id < len(PRESET_COLORS):
        return PRESET_COLORS[step_id]
    # Fallback HSV generator for high class counts
    import colorsys
    hue = ((step_id * 37) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.90)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def get_contrast_text_color(hex_color: str):
    """Determine whether white or black text is more readable against hex_color."""
    h = hex_color.lstrip("#")
    try:
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        lum = 0.299 * (r / 255.0) + 0.587 * (g / 255.0) + 0.114 * (b / 255.0)
        return "#111111" if lum > 0.58 else "#FFFFFF"
    except Exception:
        return "#111111"


# -----------------------------------------------------------------------------
# Terminal ASCII Timeline Renderer
# -----------------------------------------------------------------------------

def format_ascii_bar(label_frames, id2name, bar_width=80):
    """Render a 1D frame label sequence into an aligned ASCII timeline bar."""
    T = len(label_frames)
    if T == 0:
        return "[Empty]"

    # Subsample or bin into `bar_width` slots
    bin_labels = []
    for i in range(bar_width):
        t0 = int(i * T / bar_width)
        t1 = max(int((i + 1) * T / bar_width), t0 + 1)
        vals, counts = np.unique(label_frames[t0:t1], return_counts=True)
        bin_labels.append(vals[np.argmax(counts)])

    # Group contiguous blocks
    chunks = []
    cur_lbl = bin_labels[0]
    cur_len = 1
    for lbl in bin_labels[1:]:
        if lbl == cur_lbl:
            cur_len += 1
        else:
            chunks.append((cur_lbl, cur_len))
            cur_lbl = lbl
            cur_len = 1
    chunks.append((cur_lbl, cur_len))

    # Format each chunk with text
    parts = []
    for lbl, length in chunks:
        name = "BG" if lbl == 0 else id2name.get(lbl, f"S{lbl}")
        # Truncate or abbreviate if too long
        if length <= 2:
            cell = "." * length
        elif length <= 4:
            txt = f"{lbl}" if lbl != 0 else "·"
            cell = txt.center(length)
        else:
            # Shorten name to fit
            avail = length - 2
            txt = name[:avail]
            cell = f"[{txt.center(avail)}]"
        parts.append(cell)

    return "".join(parts)


def print_ascii_timeline_comparison(vid, duration_sec, gt_frames, paper_frames, our_frames, id2name):
    """Print a terminal ASCII timeline comparing GT, Paper Full-AEM, and Our Trained No-AEM."""
    bar_w = 75
    gt_bar = format_ascii_bar(gt_frames, id2name, bar_width=bar_w)
    our_bar = format_ascii_bar(our_frames, id2name, bar_width=bar_w) if our_frames is not None else "  (No checkpoint loaded)"
    paper_bar = format_ascii_bar(paper_frames, id2name, bar_width=bar_w) if paper_frames is not None else "  (No paper checkpoint)"

    print("\n" + "=" * 95)
    print(f" TIMELINE BAR: Video '{vid}' | Duration: {duration_sec:.1f}s ({len(gt_frames)} frames)")
    print("-" * 95)
    print(f"  GT        : {gt_bar}")
    if paper_frames is not None:
        print(f"  Paper AEM : {paper_bar}")
    if our_frames is not None:
        print(f"  Our No-AEM: {our_bar}")
    print("=" * 95)


# -----------------------------------------------------------------------------
# Matplotlib Timeline Bar Figure Generator
# -----------------------------------------------------------------------------

def plot_timeline_comparison_figure(
    vid: str,
    task_id: str,
    recipe_name: str,
    fps: float,
    gt_frames: np.ndarray,
    our_frames: np.ndarray = None,
    paper_frames: np.ndarray = None,
    id2name: dict = None,
    our_video_metrics: dict = None,
    paper_video_metrics: dict = None,
    save_path: str = None,
    display_colab: bool = True,
):
    """Generate a publication-grade stacked timeline comparison image (PNG)."""
    if not HAS_MATPLOTLIB:
        return None

    if id2name is None:
        id2name = {}

    T = len(gt_frames)
    duration_sec = T / max(fps, 1e-4)

    # Determine which bars we will plot
    bars = []
    # Order from top to bottom: GT on top, Paper in middle, Our No-AEM on bottom
    bars.append(("Ground Truth (GT)", gt_frames, None))
    if paper_frames is not None:
        bars.append(("Paper Full-AEM", paper_frames, paper_video_metrics))
    if our_frames is not None:
        bars.append(("Trained No-AEM", our_frames, our_video_metrics))

    num_bars = len(bars)
    fig_h = 2.4 + num_bars * 1.3
    fig, ax = plt.subplots(figsize=(15, fig_h), dpi=180)

    # Collect unique active action classes across all bars for the legend
    active_classes = set(np.unique(gt_frames).tolist())
    if paper_frames is not None:
        active_classes.update(np.unique(paper_frames).tolist())
    if our_frames is not None:
        active_classes.update(np.unique(our_frames).tolist())
    active_classes = sorted([int(c) for c in active_classes if c > 0])

    y_positions = list(range(num_bars - 1, -1, -1))

    for idx, (bar_title, frames, v_metrics) in enumerate(bars):
        y = y_positions[idx]
        segments = framewise_to_segments(frames)

        for lbl, start_idx, end_idx in segments:
            lbl = int(lbl)
            s_sec = start_idx / fps
            e_sec = (end_idx + 1) / fps
            w_sec = max(e_sec - s_sec, 0.05)

            color = get_step_color(lbl)
            edge_color = "#333333" if lbl == 0 else "#111111"
            alpha = 0.85 if lbl == 0 else 0.95

            # Draw segment rectangle
            ax.barh(
                y,
                w_sec,
                left=s_sec,
                height=0.62,
                color=color,
                edgecolor=edge_color,
                linewidth=0.8,
                alpha=alpha,
                align="center",
            )

            # Draw step label inside the segment if wide enough
            frac = w_sec / max(duration_sec, 1e-4)
            step_name = "BG" if lbl == 0 else id2name.get(lbl, f"Step {lbl}")

            if frac >= 0.06:
                # Segment is wide: print full / clean step name
                txt_color = get_contrast_text_color(color)
                # Shorten long text
                display_txt = step_name if len(step_name) <= 24 else step_name[:22] + "…"
                if lbl != 0:
                    display_txt = f"{lbl}. {display_txt}"
                ax.text(
                    s_sec + w_sec / 2.0,
                    y,
                    display_txt,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=txt_color,
                    clip_on=True,
                )
            elif frac >= 0.025 and lbl != 0:
                # Narrow segment: print step number
                txt_color = get_contrast_text_color(color)
                ax.text(
                    s_sec + w_sec / 2.0,
                    y,
                    str(lbl),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                    color=txt_color,
                    clip_on=True,
                )

        # Build Y-axis label with metrics if available
        label_str = bar_title
        if v_metrics is not None:
            acc = v_metrics.get("acc", 0.0)
            edit = v_metrics.get("edit", 0.0)
            frag = v_metrics.get("frag", 1.0)
            label_str += f"\n(Acc: {acc:.1f}% | Edit: {edit:.1f}% | Frag: {frag:.2f}x)"
        bars[idx] = (label_str, frames, v_metrics)

    # Configure axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels([b[0] for b in bars], fontsize=9.5, fontweight="semibold")
    ax.set_xlim(0, duration_sec)
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="medium", labelpad=6)

    # Format X ticks with both seconds and MM:SS
    ticks = np.linspace(0, duration_sec, num=min(int(duration_sec // 15) + 1, 10))
    if len(ticks) < 4:
        ticks = np.linspace(0, duration_sec, num=5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}s\n({int(t//60):02d}:{int(t%60):02d})" for t in ticks], fontsize=8.5)

    # Add subtitle & title
    m, s = divmod(int(duration_sec), 60)
    fig.suptitle(
        f"Task {task_id}: {recipe_name} | Video: '{vid}' (Duration: {duration_sec:.1f}s / {m:02d}:{s:02d})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # Build Legend of Action Steps at the bottom
    legend_handles = []
    # Add Background legend
    legend_handles.append(
        mpatches.Patch(facecolor=get_step_color(0), edgecolor="#666666", label="0: Background")
    )
    for c in active_classes:
        name = id2name.get(c, f"Step {c}")
        short_name = name if len(name) <= 30 else name[:28] + "…"
        legend_handles.append(
            mpatches.Patch(facecolor=get_step_color(c), edgecolor="#222222", label=f"{c}: {short_name}")
        )

    # Place legend below the plot with clean wrapping
    ncol = min(len(legend_handles), 4)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        fontsize=8.0,
        frameon=True,
        fancybox=True,
        facecolor="#FBFBFB",
        edgecolor="#D0D0D0",
    )

    plt.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.30)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [Saved Timeline Image] -> {save_path}")

    # Display directly in Google Colab / Jupyter cell output
    if display_colab:
        try:
            from IPython.display import Image, display
            display(Image(filename=save_path))
        except Exception:
            pass

    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# Video Evaluation & Segment Calculation
# -----------------------------------------------------------------------------

def compute_video_level_metrics(gt_frames, pred_frames):
    """Compute frame-level accuracy, Levenshtein edit, and segment count for one video."""
    min_len = min(len(gt_frames), len(pred_frames))
    yt = gt_frames[:min_len]
    yp = pred_frames[:min_len]

    acc = (yt == yp).mean() * 100.0 if min_len > 0 else 0.0

    gt_segs = framewise_to_segments(yt)
    pred_segs = framewise_to_segments(yp)

    # Segment label sequences (ignoring background=0 for edit calculation)
    gt_seq = [s[0] for s in gt_segs if s[0] != 0]
    pred_seq = [s[0] for s in pred_segs if s[0] != 0]

    # Levenshtein distance
    m, n = len(gt_seq), len(pred_seq)
    if m == 0 and n == 0:
        edit = 100.0
    elif m == 0 or n == 0:
        edit = 0.0
    else:
        dp = np.zeros((m + 1, n + 1), dtype=int)
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt_seq[i - 1] == pred_seq[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        edit = max(0.0, (1.0 - dp[m, n] / max(m, n)) * 100.0)

    frag = len(pred_segs) / max(len(gt_segs), 1)

    return {
        "acc": acc,
        "edit": edit,
        "frag": frag,
        "gt_segs_count": len(gt_segs),
        "pred_segs_count": len(pred_segs),
    }


# -----------------------------------------------------------------------------
# Main Evaluation Loop
# -----------------------------------------------------------------------------

def evaluate_and_compare_task(
    args,
    task_id: str,
    recipe_name: str,
    clip_model,
    activity_step_collection,
    step_id_mapping,
    split_file,
):
    """Run evaluation and timeline generation for one recipe comparing Our No-AEM vs. Paper Full-AEM."""
    if not os.path.exists(args.config):
        for candidate in [
            os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"),
            os.path.join(REPO_ROOT, "configs", os.path.basename(args.config)),
            "configs/ccp4d.yaml",
        ]:
            if os.path.exists(candidate):
                args.config = candidate
                break

    cfg = load_config(args.config)
    cfg["dataset"]["task"] = str(task_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["devices"] = [device]
    root_dir = _resolve_root_dir(cfg)
    fps = cfg["dataset"].get("default_fps", 10.0)

    # 1. Prepare Test Dataset
    allowed_vids = load_split_video_ids(split_file, subset=args.split_subset, task=str(task_id))
    test_dataset = CombinedCaptainCook4DDataset(
        root_dir=root_dir,
        task=str(task_id),
        features_subdir=cfg["dataset"].get("features_subdir", "I3D"),
        default_fps=fps,
        allowed_vids=allowed_vids,
    )
    if len(test_dataset) == 0:
        print(f"[Warning] Task {task_id} ({recipe_name}): No test videos found. Skipping.")
        return None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["loader"].get("num_workers", 2),
        collate_fn=lambda b: b,
    )

    # 2. Tokenize Actions & Build Name Map
    action_tokens, num_classes = action_tokenize_ccp4d(
        clip_model, str(task_id), activity_step_collection, step_id_mapping, device=device
    )
    cfg["model"]["num_classes"] = num_classes
    id2name = build_action_name_map(cfg)

    # 3. Locate Trained No-AEM Checkpoint
    our_ckpt = args.our_ckpt
    if our_ckpt is None:
        candidates = sorted(glob.glob(os.path.join(args.our_ckpt_dir, f"recipe_{task_id}_*.pth.tar")))
        if candidates:
            our_ckpt = candidates[0]
        else:
            # Fallback search
            alt = sorted(glob.glob(f"checkpoints/**/recipe_{task_id}_*no_aem*.pth.tar", recursive=True))
            if alt:
                our_ckpt = alt[0]

    # 4. Locate Paper Checkpoint
    paper_ckpt = None
    paper_candidates = sorted(glob.glob(os.path.join(args.paper_ckpt_dir, f"recipe_{task_id}_*.pth.tar")))
    if paper_candidates:
        paper_ckpt = paper_candidates[0]

    print("\n" + "#" * 105)
    print(f" EVALUATING TASK {task_id:>2s}: {recipe_name} ({len(test_dataset)} test videos)")
    print(f"   - Our Checkpoint   : {our_ckpt if our_ckpt else '[NOT FOUND]'}")
    print(f"   - Paper Checkpoint : {paper_ckpt if paper_ckpt else '[NOT FOUND]'}")
    print("#" * 105)

    # 5. Run Inference on Our Trained No-AEM Model
    our_results = None
    our_metrics = None
    if our_ckpt and os.path.exists(our_ckpt):
        model_our = make_meta_arch(cfg["model_name"], **cfg["model"])
        if torch.cuda.is_available() and len(cfg["devices"]) > 1:
            model_our = nn.DataParallel(model_our, device_ids=cfg["devices"])
        model_our = model_our.to(device)
        model_our = load_weights(model_our, our_ckpt, use_ema=False)
        model_our = configure_no_aem_model(model_our)

        t0 = time.time()
        if args.eval_ed:
            our_results = run_full_inference(model_our, test_loader, clip_model, action_tokens)
            our_ed = compute_ed_metrics(our_results)
        else:
            our_results = run_segmentation_only(model_our, test_loader)
            our_ed = {"precision": 0.0, "auc": 0.0, "eda": 0.0}

        gt_frames_list = [v["frame_action_label"].cpu().numpy() for v in our_results.values()]
        our_pred_frames = [v["frame_action_pred"].cpu().numpy() for v in our_results.values()]
        our_w = compute_extended_seg_metrics(gt_frames_list, our_pred_frames, ignore_bg=False)
        our_metrics = {"with_bg": our_w, "ed": our_ed, "time": time.time() - t0}
        del model_our

    # 6. Run Inference on Paper Released Model
    paper_results = None
    paper_metrics = None
    if paper_ckpt and os.path.exists(paper_ckpt):
        model_pap = make_meta_arch(cfg["model_name"], **cfg["model"])
        if torch.cuda.is_available() and len(cfg["devices"]) > 1:
            model_pap = nn.DataParallel(model_pap, device_ids=cfg["devices"])
        model_pap = model_pap.to(device)
        model_pap = load_weights(model_pap, paper_ckpt, use_ema=False)

        t0 = time.time()
        if args.eval_ed:
            paper_results = run_full_inference(model_pap, test_loader, clip_model, action_tokens)
            paper_ed = compute_ed_metrics(paper_results)
        else:
            paper_results = run_segmentation_only(model_pap, test_loader)
            paper_ed = {"precision": 0.0, "auc": 0.0, "eda": 0.0}

        gt_frames_list = [v["frame_action_label"].cpu().numpy() for v in paper_results.values()]
        pap_pred_frames = [v["frame_action_pred"].cpu().numpy() for v in paper_results.values()]
        pap_w = compute_extended_seg_metrics(gt_frames_list, pap_pred_frames, ignore_bg=False)
        paper_metrics = {"with_bg": pap_w, "ed": paper_ed, "time": time.time() - t0}
        del model_pap

    # 7. Generate Timeline Bars (ASCII + PNG) for Selected Test Videos
    ref_results = our_results or paper_results
    if ref_results is not None:
        vis_count = 0
        max_vis = len(ref_results) if args.max_vis_videos < 0 else args.max_vis_videos

        for vid, ref_v in ref_results.items():
            if vis_count >= max_vis:
                break
            vis_count += 1

            gt_f = ref_v["frame_action_label"].cpu().numpy()
            our_f = our_results[vid]["frame_action_pred"].cpu().numpy() if our_results else None
            pap_f = paper_results[vid]["frame_action_pred"].cpu().numpy() if paper_results else None

            our_v_m = compute_video_level_metrics(gt_f, our_f) if our_f is not None else None
            pap_v_m = compute_video_level_metrics(gt_f, pap_f) if pap_f is not None else None

            # Print ASCII timeline bar
            if args.ascii:
                duration_sec = len(gt_f) / max(fps, 1e-4)
                print_ascii_timeline_comparison(vid, duration_sec, gt_f, pap_f, our_f, id2name)
                if our_v_m:
                    print(f"  -> Our No-AEM : Acc: {our_v_m['acc']:5.2f}% | Edit: {our_v_m['edit']:5.2f}% | Segments: {our_v_m['pred_segs_count']} (Frag: {our_v_m['frag']:.2f}x)")
                if pap_v_m:
                    print(f"  -> Paper AEM  : Acc: {pap_v_m['acc']:5.2f}% | Edit: {pap_v_m['edit']:5.2f}% | Segments: {pap_v_m['pred_segs_count']} (Frag: {pap_v_m['frag']:.2f}x)")

            # Generate PNG Timeline Bar Image
            if args.save_png and HAS_MATPLOTLIB:
                clean_recipe = "".join(c for c in recipe_name if c.isalnum() or c in ("_", "-"))
                png_path = os.path.join(
                    args.output_dir, f"timeline_task_{task_id}_{clean_recipe}_{vid}.png"
                )
                plot_timeline_comparison_figure(
                    vid=vid,
                    task_id=task_id,
                    recipe_name=recipe_name,
                    fps=fps,
                    gt_frames=gt_f,
                    our_frames=our_f,
                    paper_frames=pap_f,
                    id2name=id2name,
                    our_video_metrics=our_v_m,
                    paper_video_metrics=pap_v_m,
                    save_path=png_path,
                    display_colab=args.display_colab,
                )

    # 8. Print Task-Level Comparison Summary
    print("\n" + "-" * 115)
    print(f" TASK {task_id} SUMMARY: {recipe_name}")
    print("-" * 115)
    if our_metrics:
        w = our_metrics["with_bg"]
        ed = our_metrics["ed"]
        print(
            f"  [Trained No-AEM] Acc: {w['Accuracy']:5.2f}% | Edit: {w['Edit']:5.2f}% | "
            f"F1@50: {w['F1@50']:5.2f}% | Frag: {w['frag_ratio']:4.2f}x || "
            f"Prec: {ed['precision']:5.2f}% | AUC: {ed['auc']:5.2f}% | EDA: {ed['eda']:5.2f}%"
        )
    if paper_metrics:
        w = paper_metrics["with_bg"]
        ed = paper_metrics["ed"]
        print(
            f"  [Paper Full-AEM] Acc: {w['Accuracy']:5.2f}% | Edit: {w['Edit']:5.2f}% | "
            f"F1@50: {w['F1@50']:5.2f}% | Frag: {w['frag_ratio']:4.2f}x || "
            f"Prec: {ed['precision']:5.2f}% | AUC: {ed['auc']:5.2f}% | EDA: {ed['eda']:5.2f}%"
        )
    print("-" * 115)

    return {
        "task_id": task_id,
        "recipe": recipe_name,
        "our": our_metrics,
        "paper": paper_metrics,
    }


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Timeline Bar Comparison & Evaluation: Trained No-AEM vs. Paper Full-AEM"
    )
    parser.add_argument("--tasks", type=str, default="1",
                        help="Task ID(s) to evaluate: '1', '1,2,3', or 'all' (default: 1)")
    parser.add_argument("--our_ckpt", type=str, default=None,
                        help="Direct path to a specific trained No-AEM checkpoint file")
    parser.add_argument("--our_ckpt_dir", type=str,
                        default="/content/drive/MyDrive/AEM/checkpoints/ccp4d_no_aem",
                        help="Directory containing trained checkpoints (recipe_{task_id}_*.pth.tar)")
    parser.add_argument("--paper_ckpt_dir", type=str,
                        default=os.path.join(REPO_ROOT, "checkpoints", "ccp4d"),
                        help="Directory containing author released checkpoints")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"),
                        help="Base ActionFormer configuration file (default: configs/ccp4d.yaml)")
    parser.add_argument("--split_file", type=str, default=DEFAULT_SPLIT_FILE,
                        help="Path to CaptainCook4D split file")
    parser.add_argument("--split_subset", type=str, default="test", choices=["train", "val", "test"],
                        help="Split subset to evaluate on (default: test)")
    parser.add_argument("--output_dir", type=str, default="timeline_plots",
                        help="Directory to save timeline PNG images (default: timeline_plots)")
    parser.add_argument("--max_vis_videos", type=int, default=3,
                        help="Max test videos per recipe to generate timeline bars for (-1 for all)")
    parser.add_argument("--eval_ed", action="store_true", default=True,
                        help="Evaluate Mistake Detection (Precision/AUC/EDA) using OpenCLIP")
    parser.add_argument("--no_eval_ed", action="store_false", dest="eval_ed",
                        help="Skip OpenCLIP mistake detection for ultra-fast pure segmentation testing")
    parser.add_argument("--ascii", action="store_true", default=True,
                        help="Print ASCII timeline bars in terminal (default: True)")
    parser.add_argument("--no_ascii", action="store_false", dest="ascii",
                        help="Disable ASCII timeline output")
    parser.add_argument("--save_png", action="store_true", default=True,
                        help="Save high-resolution matplotlib timeline PNG images (default: True)")
    parser.add_argument("--no_png", action="store_false", dest="save_png",
                        help="Disable PNG image saving")
    parser.add_argument("--display_colab", action="store_true", default=True,
                        help="Display PNG inline in Google Colab / Jupyter notebook (default: True)")
    return parser.parse_args()


def main():
    args = parse_args()
    _ = fix_random_seed(0, include_cuda=True)

    if not os.path.exists(args.config):
        for candidate in [
            os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"),
            os.path.join(REPO_ROOT, "configs", os.path.basename(args.config)),
            "configs/ccp4d.yaml",
        ]:
            if os.path.exists(candidate):
                args.config = candidate
                break

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["devices"] = [device]
    root_dir = _resolve_root_dir(cfg)

    # Load CaptainCook4D task mappings
    id_activity_file = os.path.join(root_dir, "id_activity_mapping.json")
    with open(id_activity_file, "r") as fp:
        id_activity = json.load(fp)

    with open(os.path.join(root_dir, "activity_step_collection.json"), "r") as fp:
        activity_step_collection = json.load(fp)

    with open(os.path.join(root_dir, "step_id_mapping.json"), "r") as fp:
        step_id_mapping = json.load(fp)

    # Parse task list
    t_arg = args.tasks.strip().lower()
    if t_arg == "all":
        task_list = [(str(tid), rname) for tid, rname in id_activity.items() if str(tid) != "0"]
    else:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        task_list = [(t, id_activity[t]) for t in wanted if t in id_activity]

    if not task_list:
        raise ValueError(f"No valid tasks matched --tasks={args.tasks}")

    print("=" * 115)
    print(f" TIMELINE BAR TEST & COMPARISON | Tasks: {len(task_list)} | Subset: {args.split_subset}")
    print(f" Output Directory: {args.output_dir} | Max Vis Videos / Task: {args.max_vis_videos}")
    print("=" * 115)

    clip_model = load_clip_model(device) if args.eval_ed else None

    all_results = []
    for tid, rname in task_list:
        res = evaluate_and_compare_task(
            args, tid, rname, clip_model, activity_step_collection, step_id_mapping, args.split_file
        )
        if res is not None:
            all_results.append(res)

    if not all_results:
        return

    # Print Final Macro Table across evaluated tasks
    print("\n" + "=" * 136)
    print(f" FINAL SIDE-BY-SIDE SUMMARY TABLE ({len(all_results)} TASKS)")
    print("=" * 136)
    print(
        f"{'Task':<5} {'Recipe':<30} | {'Model Variant':<16} | "
        f"{'Acc(w/BG)':>9} {'Edit':>6} {'F1@50':>6} {'Frag':>6} | "
        f"{'Prec':>6} {'AUC':>6} {'EDA':>6}"
    )
    print("-" * 136)

    our_w_list, our_ed_list = [], []
    pap_w_list, pap_ed_list = [], []

    for r in all_results:
        tid = r["task_id"]
        rname = r["recipe"][:30]

        if r["our"] is not None:
            ow = r["our"]["with_bg"]
            oed = r["our"]["ed"]
            our_w_list.append(ow)
            our_ed_list.append(oed)
            print(
                f"{tid:<5} {rname:<30} | {'Trained No-AEM':<16} | "
                f"{ow['Accuracy']:>8.2f}% {ow['Edit']:>6.2f} {ow['F1@50']:>6.2f} {ow['frag_ratio']:>5.2f}x | "
                f"{oed['precision']:>6.2f} {oed['auc']:>6.2f} {oed['eda']:>6.2f}"
            )

        if r["paper"] is not None:
            pw = r["paper"]["with_bg"]
            ped = r["paper"]["ed"]
            pap_w_list.append(pw)
            pap_ed_list.append(ped)
            print(
                f"{'':<5} {'':<30} | {'Paper Full-AEM':<16} | "
                f"{pw['Accuracy']:>8.2f}% {pw['Edit']:>6.2f} {pw['F1@50']:>6.2f} {pw['frag_ratio']:>5.2f}x | "
                f"{ped['precision']:>6.2f} {ped['auc']:>6.2f} {ped['eda']:>6.2f}"
            )
        print("-" * 136)

    # Print Macro Average if multiple recipes evaluated
    if len(our_w_list) > 1:
        acc = np.mean([x["Accuracy"] for x in our_w_list])
        edit = np.mean([x["Edit"] for x in our_w_list])
        f1 = np.mean([x["F1@50"] for x in our_w_list])
        frag = sum(x["pred_segs"] for x in our_w_list) / max(1, sum(x["gt_segs"] for x in our_w_list))
        prec = np.mean([x["precision"] for x in our_ed_list])
        auc = np.mean([x["auc"] for x in our_ed_list])
        eda = np.mean([x["eda"] for x in our_ed_list])
        print(
            f"{'AVG':<5} {'Macro Average (Ours)':<30} | {'Trained No-AEM':<16} | "
            f"{acc:>8.2f}% {edit:>6.2f} {f1:>6.2f} {frag:>5.2f}x | "
            f"{prec:>6.2f} {auc:>6.2f} {eda:>6.2f}"
        )
    if len(pap_w_list) > 1:
        acc = np.mean([x["Accuracy"] for x in pap_w_list])
        edit = np.mean([x["Edit"] for x in pap_w_list])
        f1 = np.mean([x["F1@50"] for x in pap_w_list])
        frag = sum(x["pred_segs"] for x in pap_w_list) / max(1, sum(x["gt_segs"] for x in pap_w_list))
        prec = np.mean([x["precision"] for x in pap_ed_list])
        auc = np.mean([x["auc"] for x in pap_ed_list])
        eda = np.mean([x["eda"] for x in pap_ed_list])
        print(
            f"{'AVG':<5} {'Macro Average (Paper)':<30} | {'Paper Full-AEM':<16} | "
            f"{acc:>8.2f}% {edit:>6.2f} {f1:>6.2f} {frag:>5.2f}x | "
            f"{prec:>6.2f} {auc:>6.2f} {eda:>6.2f}"
        )
        print("=" * 136)


if __name__ == "__main__":
    main()
