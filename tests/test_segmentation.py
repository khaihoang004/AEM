"""Standalone evaluation and video visualization script for Action Segmentation.

Supports:
1. Evaluating ONLY the Action Segmentation branch (no OpenCLIP required).
2. Filtering test videos using CaptainCook4D data split files (e.g.,
   `person_data_split_combined.json` from CaptainCook4D/annotations, combining
   both normal and error recordings according to `--split_subset` train/val/test).
3. Exporting annotated visualization videos (`.mp4`) with a top/bottom banner and
   dual timeline bar comparing Ground-Truth vs. Predicted action segments.
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Ensure repo root is on sys.path when invoked from anywhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.datasets.clip_encode import action_tokenize_egoper
from libs.datasets.clip_encode_ccp4d import action_tokenize_ccp4d
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from metric import calculate_micro_auc, evaluate_ed, framewise_to_segments

warnings.filterwarnings("ignore")

DEFAULT_SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person_data_split_combined.json")
F1_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
ED_THRESHOLDS = np.arange(0.0, 1.0, 0.025)


def load_clip_model(device="cuda"):
    """Load the frozen EVA-02 CLIP used for prompt / mistake-detection scoring."""
    import open_clip

    clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    return (clip.eval().requires_grad_(False).to(device), tokenizer, preprocess)


def compute_ed_metrics(results):
    """Compute Mistake Detection metrics (EDA, Precision, Micro AUC) across the 40 thresholds."""
    eda_list, precision_list, micro_fprs, micro_tprs = [], [], [], []
    for threshold in ED_THRESHOLDS:
        outputs = evaluate_ed(results, threshold=float(threshold))
        eda_list.append(outputs["error_acc"])
        precision_list.append(outputs["micro"]["precision"])
        micro_fprs.append(outputs["micro"]["fpr"])
        micro_tprs.append(outputs["micro"]["tpr"])
    return {
        "eda": float(np.mean(eda_list)) * 100.0,
        "precision": float(np.mean(precision_list)) * 100.0,
        "auc": float(calculate_micro_auc(micro_fprs, micro_tprs)) * 100.0,
    }


def compute_extended_seg_metrics(list_of_y_true, list_of_y_pred, ignore_bg=False, bg_class=0):
    """Compute comprehensive Action Segmentation metrics (with or without Background class).

    Metrics returned (percentages in [0, 100], except segment counts & ratio):
      - Accuracy: micro frame accuracy (ignoring frames where y_true == bg_class if ignore_bg=True)
      - mIoU: macro average across videos of per-video mean class IoU
      - Edit: macro average across videos of normalized Levenshtein edit score
      - F1@10, F1@25, F1@50, F1@75 (and alias F1@0.5): macro average across videos
      - pred_segs: total predicted segments across all evaluated videos
      - gt_segs: total ground-truth segments across all evaluated videos
      - frag_ratio: fragmentation / over-segmentation ratio (#Pred / #GT)
    """
    total_correct = 0
    total_frames = 0
    ious = []
    edits = []
    f1_by_tau = {tau: [] for tau in F1_THRESHOLDS}
    total_pred_segs = 0
    total_gt_segs = 0

    for y_true, y_pred in zip(list_of_y_true, list_of_y_pred):
        min_len = min(len(y_true), len(y_pred))
        yt = np.asarray(y_true[:min_len])
        yp = np.asarray(y_pred[:min_len])

        # 1. Frame Accuracy
        if ignore_bg:
            valid_mask = yt != bg_class
            total_correct += int(np.sum(yt[valid_mask] == yp[valid_mask]))
            total_frames += int(np.sum(valid_mask))
        else:
            total_correct += int(np.sum(yt == yp))
            total_frames += len(yt)

        # 2. Per-video mIoU
        all_classes = np.unique(np.concatenate((yt, yp)))
        if ignore_bg:
            all_classes = [c for c in all_classes if c != bg_class]
        inter_sum, valid_cls = 0.0, 0
        for cls in all_classes:
            inter = np.sum((yt == cls) & (yp == cls))
            uni = np.sum((yt == cls) | (yp == cls))
            if uni > 0:
                inter_sum += inter / uni
                valid_cls += 1
        ious.append((inter_sum / valid_cls) if valid_cls > 0 else 0.0)

        # Extract contiguous segments: list of (label, start, end)
        gt_segs = framewise_to_segments(yt)
        pred_segs = framewise_to_segments(yp)
        if ignore_bg:
            gt_segs = [s for s in gt_segs if s[0] != bg_class]
            pred_segs = [s for s in pred_segs if s[0] != bg_class]

        total_gt_segs += len(gt_segs)
        total_pred_segs += len(pred_segs)

        # 3. Edit Score (Levenshtein on segment label sequence)
        t_seq = [s[0] for s in gt_segs]
        p_seq = [s[0] for s in pred_segs]
        len_t, len_p = len(t_seq), len(p_seq)
        if len_t == 0 and len_p == 0:
            edits.append(1.0)
        elif max(len_t, len_p) == 0:
            edits.append(0.0)
        else:
            D = np.zeros((len_p + 1, len_t + 1), dtype=int)
            D[:, 0] = np.arange(len_p + 1)
            D[0, :] = np.arange(len_t + 1)
            for i in range(1, len_p + 1):
                for j in range(1, len_t + 1):
                    if p_seq[i - 1] == t_seq[j - 1]:
                        D[i, j] = D[i - 1, j - 1]
                    else:
                        D[i, j] = 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            edits.append(1.0 - D[len_p, len_t] / max(len_p, len_t))

        # 4. Segment-level F1@{10, 25, 50, 75}
        for tau in F1_THRESHOLDS:
            matched_gt = set()
            tp = 0
            for p_lbl, p_s, p_e in pred_segs:
                best_iou = 0.0
                best_g_idx = None
                for g_idx, (g_lbl, g_s, g_e) in enumerate(gt_segs):
                    if g_idx in matched_gt or g_lbl != p_lbl:
                        continue
                    inter = max(0, min(p_e, g_e) - max(p_s, g_s) + 1)
                    uni = (p_e - p_s + 1) + (g_e - g_s + 1) - inter
                    iou = inter / uni if uni > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
                if best_iou >= tau and best_g_idx is not None:
                    tp += 1
                    matched_gt.add(best_g_idx)

            fp = len(pred_segs) - tp
            fn = len(gt_segs) - len(matched_gt)
            if (tp + fp) == 0 or (tp + fn) == 0 or tp == 0:
                f1_by_tau[tau].append(0.0)
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1_by_tau[tau].append(2.0 * prec * rec / (prec + rec))

    acc = (total_correct / total_frames * 100.0) if total_frames > 0 else 0.0
    mIoU = float(np.mean(ious)) * 100.0 if ious else 0.0
    edit = float(np.mean(edits)) * 100.0 if edits else 0.0
    f1_10 = float(np.mean(f1_by_tau[0.10])) * 100.0 if f1_by_tau[0.10] else 0.0
    f1_25 = float(np.mean(f1_by_tau[0.25])) * 100.0 if f1_by_tau[0.25] else 0.0
    f1_50 = float(np.mean(f1_by_tau[0.50])) * 100.0 if f1_by_tau[0.50] else 0.0
    f1_75 = float(np.mean(f1_by_tau[0.75])) * 100.0 if f1_by_tau[0.75] else 0.0
    frag_ratio = (total_pred_segs / total_gt_segs) if total_gt_segs > 0 else 0.0

    return {
        "Accuracy": acc,
        "mIoU": mIoU,
        "Edit": edit,
        "F1@10": f1_10,
        "F1@25": f1_25,
        "F1@50": f1_50,
        "F1@75": f1_75,
        "F1@0.5": f1_50,
        "pred_segs": int(total_pred_segs),
        "gt_segs": int(total_gt_segs),
        "frag_ratio": float(frag_ratio),
    }

def resolve_ckpt_file(ckpt, epoch=-1, is_best=False):
    """Resolve a checkpoint path, which may be a file or a folder of epoch checkpoints."""
    if ".pth.tar" in ckpt:
        assert os.path.isfile(ckpt), f"CKPT file does not exist: {ckpt}"
        return ckpt
    assert os.path.isdir(ckpt), f"CKPT folder does not exist: {ckpt}"
    if epoch > 0:
        ckpt_file = os.path.join(ckpt, f"epoch_{epoch:03d}.pth.tar")
    elif is_best:
        ckpt_file = os.path.join(ckpt, "best_val.pth.tar")
    else:
        candidates = sorted(glob.glob(os.path.join(ckpt, "*.pth.tar")))
        assert len(candidates) > 0, f"No .pth.tar files found in {ckpt}"
        ckpt_file = candidates[-1]
    assert os.path.exists(ckpt_file), f"Resolved CKPT file does not exist: {ckpt_file}"
    return ckpt_file


def load_weights(model, ckpt_file, use_ema=False):
    """Load checkpoint weights into `model`."""
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    if use_ema:
        assert "state_dict_ema" in checkpoint, (
            "--use_ema requested but this checkpoint has no 'state_dict_ema' entry."
        )
        state_dict, source = checkpoint["state_dict_ema"], "EMA weights"
    else:
        state_dict, source = checkpoint["state_dict"], "model weights"

    wrapped = isinstance(model, nn.DataParallel)
    normalized = OrderedDict()
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        normalized["module." + k if wrapped else k] = v

    model.load_state_dict(normalized)
    print(f"=> loaded {source} from '{ckpt_file}' ({len(normalized)} tensors)")
    del checkpoint
    return model


def _resolve_root_dir(cfg):
    """Ensure cfg['dataset']['root_dir'] resolves relative to REPO_ROOT if not found from cwd."""
    root_dir = cfg["dataset"]["root_dir"]
    if not os.path.isabs(root_dir) and not os.path.exists(root_dir):
        candidate = os.path.join(REPO_ROOT, root_dir)
        if os.path.exists(candidate):
            cfg["dataset"]["root_dir"] = candidate
    return cfg["dataset"]["root_dir"]


def build_action_name_map(cfg):
    """Return {action_id: action_name} for human-readable segment inspection."""
    root_dir = _resolve_root_dir(cfg)
    task = str(cfg["dataset"]["task"])
    id2name = {0: "BG (background)"}

    if cfg["dataset_name"] == "CaptainCook4D":
        with open(os.path.join(root_dir, "activity_step_collection.json"), "r") as fp:
            activity_step_collection = json.load(fp)
        with open(os.path.join(root_dir, "step_id_mapping.json"), "r") as fp:
            step_id_mapping = json.load(fp)
        task_map = step_id_mapping["step_id_mapping"][task]
        for activity, steps in activity_step_collection.items():
            for org_step_id, step_name in steps.items():
                if org_step_id in task_map:
                    mapped_id = task_map[org_step_id]
                    id2name[mapped_id] = step_name
    else:
        annot_file = os.path.join(root_dir, "annotation.json")
        if os.path.isfile(annot_file):
            with open(annot_file, "r") as fp:
                all_annot = json.load(fp)
            idx_to_act = all_annot.get(task, {}).get("idx_to_action", {})
            for k, v in idx_to_act.items():
                id2name[int(k)] = v

    return id2name


def load_split_video_ids(split_file, subset="test", task=None):
    """Parse a local CaptainCook4D split JSON (e.g. tests/person_data_split_combined.json)."""
    assert os.path.isfile(split_file), f"Split file does not exist: {split_file}"
    with open(split_file, "r") as fp:
        data = json.load(fp)

    def _norm_vid(v):
        v = str(v).strip()
        if v.endswith("_360p.mp4"):
            v = v[: -len("_360p.mp4")]
        elif v.endswith(".mp4"):
            v = v[: -len(".mp4")]
        elif v.endswith("_360p"):
            v = v[: -len("_360p")]
        return v

    subsets = [s.strip() for s in subset.split(",")]
    allowed = set()

    # Case 1: top-level keys are 'train', 'val', 'test'
    if any(k in data for k in ("train", "val", "validation", "test")):
        for s in subsets:
            key = "val" if (s == "val" and "val" in data) else ("validation" if s == "val" else s)
            val = data.get(key, [])
            if isinstance(val, list):
                allowed.update(_norm_vid(x) for x in val)
            elif isinstance(val, dict):
                if task is not None and str(task) in val:
                    allowed.update(_norm_vid(x) for x in val[str(task)])
                else:
                    for sub_list in val.values():
                        if isinstance(sub_list, list):
                            allowed.update(_norm_vid(x) for x in sub_list)
    # Case 2: top-level keys are recipe IDs ("1", "2", ...) or recording IDs
    elif task is not None and str(task) in data and isinstance(data[str(task)], dict):
        rec_dict = data[str(task)]
        for s in subsets:
            for x in rec_dict.get(s, []):
                allowed.add(_norm_vid(x))
    else:
        # Fallback: recursively collect lists under matching subset keys
        for k, v in data.items():
            if isinstance(v, dict):
                for s in subsets:
                    if s in v and isinstance(v[s], list):
                        allowed.update(_norm_vid(x) for x in v[s])
            elif isinstance(v, str) and v in subsets:
                allowed.add(_norm_vid(k))

    return allowed


class CombinedCaptainCook4DDataset(Dataset):
    """Dataset that pools BOTH normal and error samples of a CaptainCook4D recipe,
    filtered by an optional set of recording IDs (e.g. from person_data_split_combined.json).
    """

    def __init__(self, root_dir, task, features_subdir="I3D", default_fps=10, allowed_vids=None):
        self.feat_root_dir = os.path.join(root_dir, f"{features_subdir}_features_10fps")
        self.default_fps = default_fps
        self.task = str(task)

        with open(os.path.join(root_dir, "non_error_samples_processed.json"), "r") as f:
            non_err = json.load(f).get(self.task, {})
        with open(os.path.join(root_dir, "error_samples_processed.json"), "r") as f:
            err = json.load(f).get(self.task, {})

        merged = {}
        merged.update(non_err)
        merged.update(err)

        self.annots = []
        for video_id, video_data in merged.items():
            temp_video_id = video_id if self.task != "0" else video_id.split("_", 1)[1]
            if allowed_vids is not None and temp_video_id not in allowed_vids:
                continue
            feat_path = os.path.join(self.feat_root_dir, f"{temp_video_id}_360p.npy")
            if not os.path.exists(feat_path):
                continue
            self.annots.append(
                {
                    "video_id": temp_video_id,
                    "feat_path": feat_path,
                    "segments": video_data["segments"],
                    "labels": video_data["labels"],
                    "labels_error": video_data["labels_error"],
                }
            )

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        annot = self.annots[idx]
        feats = np.load(annot["feat_path"])
        return {
            "video_id": annot["video_id"],
            "feats": torch.from_numpy(feats).transpose(0, 1).float(),
            "segments": torch.tensor(annot["segments"]).float(),
            "labels": torch.tensor(annot["labels"]).long(),
            "labels_error": torch.tensor(annot["labels_error"]).long(),
            "fps": self.default_fps,
            "duration": len(feats) / self.default_fps,
            "visual_info": None,
            "scene_graph": None,
        }


@torch.no_grad()
def run_segmentation_only(model, test_loader):
    """Run only backbone + neck + DynClsHead/DynRegHead + NMS (skip Effect & Action models)."""
    model.eval()
    pt_model = model.module if isinstance(model, nn.DataParallel) else model
    results = {}
    for video_list in test_loader:
        fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = pt_model.extract_feature(video_list)
        seg_res = pt_model.segment_video(video_list, points, fpn_masks, cls_prob_logits, out_offsets)
        results.update(seg_res)
    return results


@torch.no_grad()
def run_full_inference(model, test_loader, clip_model, action_tokens):
    """Run full AEM inference (Segmentation + Effect_Model + Action_Model) for AUC/EDA/Precision."""
    model.eval()
    pt_model = model.module if isinstance(model, nn.DataParallel) else model
    results = {}
    for video_list in test_loader:
        results.update(
            pt_model.inference(
                video_list=video_list,
                clip_model=clip_model,
                action_tokens=action_tokens,
            )
        )
    return results


def print_video_segments(results, id2name, fps=10.0, max_videos=5):
    """Print detailed predicted vs. ground-truth segments for up to `max_videos` test videos."""
    shown = 0
    for vid, v in results.items():
        if shown >= max_videos:
            break
        shown += 1
        gt_segs = v["gt_segments"].numpy() / fps
        gt_labels = v["gt_segments_labels"].numpy()
        pred_segs = v["segments"].numpy() / fps
        pred_labels = v["segment_action_pred"].numpy()

        frame_gt = v["frame_action_label"].numpy()
        frame_pred = v["frame_action_pred"].numpy()
        T = min(len(frame_gt), len(frame_pred))
        vid_acc = (frame_gt[:T] == frame_pred[:T]).mean() * 100.0

        print(
            f"\n--- Video: {vid} | Frame Acc: {vid_acc:.2f}% | "
            f"GT segs: {len(gt_labels)} | Pred segs: {len(pred_labels)} ---"
        )
        print("  [Ground Truth Segments]:")
        for (s, e), lbl in zip(gt_segs, gt_labels):
            print(f"    {s:6.1f}s - {e:6.1f}s | class {int(lbl):2d}: {id2name.get(int(lbl), 'unknown')}")
        print("  [Predicted Segments]:")
        for (s, e), lbl in zip(pred_segs, pred_labels):
            print(f"    {s:6.1f}s - {e:6.1f}s | class {int(lbl):2d}: {id2name.get(int(lbl), 'unknown')}")


def _get_color_palette(num_classes=30):
    """Generate distinct BGR colors for each action class (0 = dark gray for BG)."""
    import cv2

    palette = {0: (70, 70, 70)}
    for c in range(1, max(num_classes + 5, 40)):
        hsv = np.uint8([[[int((c * 37) % 180), 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[c] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return palette


def find_video_source(video_root, video_id):
    """Locate either a raw video file (.mp4) or a directory of extracted frames for `video_id`."""
    if not video_root or not os.path.exists(video_root):
        return None
    root = Path(video_root)
    # Direct candidates
    candidates = [
        root / f"{video_id}_360p.mp4",
        root / f"{video_id}.mp4",
        root / "videos_360p" / f"{video_id}_360p.mp4",
        root / "videos" / f"{video_id}_360p.mp4",
        root / "videos" / f"{video_id}.mp4",
        root / "gopro" / "resolution_360p" / f"{video_id}_360p.mp4",
        root / "hololens" / f"{video_id}_360p.mp4",
        root / f"{video_id}_360p",
        root / f"{video_id}",
        root / "frames_1fps" / f"{video_id}_360p",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Recursive glob fallback (depth-limited)
    for match in root.rglob(f"{video_id}*"):
        if match.suffix.lower() in (".mp4", ".avi", ".mkv") or match.is_dir():
            return str(match)
    return None


def render_segmentation_video(
    vid,
    v_res,
    id2name,
    out_path,
    video_source=None,
    feat_fps=10.0,
    out_fps=5.0,
    max_duration_sec=None,
):
    """Render an MP4 video comparing GT vs Predicted action segmentation over time.

    Works both WITH raw video / frame folder (`video_source`) and WITHOUT raw video
    (synthesizes a clean canvas with timeline + step text so you can always inspect it).
    """
    import cv2

    frame_gt = v_res["frame_action_label"].numpy()
    frame_pred = v_res["frame_action_pred"].numpy()
    T_feat = min(len(frame_gt), len(frame_pred))
    total_sec = T_feat / feat_fps
    if max_duration_sec is not None and max_duration_sec > 0:
        total_sec = min(total_sec, float(max_duration_sec))

    palette = _get_color_palette(len(id2name))

    cap = None
    frame_files = None
    src_fps = feat_fps
    W, H = 640, 360

    if video_source and os.path.isfile(video_source):
        cap = cv2.VideoCapture(video_source)
        if cap.isOpened():
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        else:
            cap = None
    elif video_source and os.path.isdir(video_source):
        frame_files = sorted(
            glob.glob(os.path.join(video_source, "*.png"))
            + glob.glob(os.path.join(video_source, "*.jpg"))
        )
        if frame_files:
            sample_im = cv2.imread(frame_files[0])
            if sample_im is not None:
                H, W = sample_im.shape[:2]

    banner_h = 130
    out_H = H + banner_h
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (W, out_H))

    # Pre-render static GT & Pred timeline bars (width = W - 80)
    bar_x0, bar_x1 = 65, W - 15
    bar_w = max(bar_x1 - bar_x0, 10)
    gt_bar = np.zeros((18, bar_w, 3), dtype=np.uint8)
    pred_bar = np.zeros((18, bar_w, 3), dtype=np.uint8)
    for x in range(bar_w):
        f_idx = min(int((x / bar_w) * T_feat), T_feat - 1)
        gt_bar[:, x] = palette.get(int(frame_gt[f_idx]), (128, 128, 128))
        pred_bar[:, x] = palette.get(int(frame_pred[f_idx]), (128, 128, 128))

    num_out_frames = int(total_sec * out_fps)
    running_correct = 0

    for step_idx in range(num_out_frames):
        t_sec = step_idx / out_fps
        feat_idx = min(int(t_sec * feat_fps), T_feat - 1)
        gt_cls = int(frame_gt[feat_idx])
        pred_cls = int(frame_pred[feat_idx])
        if gt_cls == pred_cls:
            running_correct += 1
        acc_so_far = (running_correct / (step_idx + 1)) * 100.0

        # Read or synthesize visual frame
        base_frame = None
        if cap is not None:
            target_frame_no = int(t_sec * src_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
            ok, frm = cap.read()
            if ok and frm is not None:
                base_frame = cv2.resize(frm, (W, H))
        elif frame_files:
            # CaptainCook4D extracted frames are typically 1 fps
            f_no = min(int(t_sec), len(frame_files) - 1)
            frm = cv2.imread(frame_files[f_no])
            if frm is not None:
                base_frame = cv2.resize(frm, (W, H))

        if base_frame is None:
            base_frame = np.full((H, W, 3), 30, dtype=np.uint8)
            cv2.putText(
                base_frame,
                f"Video ID: {vid} (No raw video linked - Timeline Mode)",
                (20, H // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Build HUD banner at bottom
        banner = np.full((banner_h, W, 3), 20, dtype=np.uint8)
        match_color = (80, 220, 80) if gt_cls == pred_cls else (70, 70, 240)
        status_str = "MATCH" if gt_cls == pred_cls else "MISMATCH"

        gt_name = id2name.get(gt_cls, f"Class {gt_cls}")
        pred_name = id2name.get(pred_cls, f"Class {pred_cls}")

        cv2.putText(
            banner,
            f"Vid: {vid} | Time: {t_sec:5.1f}s / {T_feat/feat_fps:5.1f}s | [{status_str}] | Acc: {acc_so_far:5.1f}%",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            match_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            banner,
            f"GT   [{gt_cls:2d}]: {gt_name[:68]}",
            (12, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            banner,
            f"Pred [{pred_cls:2d}]: {pred_name[:68]}",
            (12, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            match_color,
            1,
            cv2.LINE_AA,
        )

        # Draw timeline bars
        cv2.putText(banner, "GT", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        banner[78:96, bar_x0:bar_x1] = gt_bar
        cv2.putText(banner, "Pred", (12, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        banner[102:120, bar_x0:bar_x1] = pred_bar

        # Draw playhead cursor on timeline bars
        cursor_x = bar_x0 + min(int((feat_idx / max(T_feat, 1)) * bar_w), bar_w - 1)
        cv2.line(banner, (cursor_x, 74), (cursor_x, 124), (255, 255, 255), 2)

        combined = np.vstack([base_frame, banner])
        writer.write(combined)

    writer.release()
    if cap is not None:
        cap.release()
    print(f"  [Video Saved] {out_path} ({num_out_frames} frames @ {out_fps} fps)")


def evaluate_task(
    cfg,
    ckpt_file,
    clip_model=None,
    eval_ed=True,
    use_gcn=False,
    use_ema=False,
    show_videos=0,
    save_json=None,
    split_file=None,
    split_subset="test",
    video_root=None,
    export_video_dir=None,
    max_export_videos=2,
    out_fps=5.0,
):
    """Evaluate segmentation AND mistake detection (Precision/AUC/EDA) on one task/recipe."""
    _ = fix_random_seed(0, include_cuda=True)
    root_dir = _resolve_root_dir(cfg)

    task = str(cfg["dataset"]["task"])
    if split_file and cfg["dataset_name"] == "CaptainCook4D":
        allowed_vids = load_split_video_ids(split_file, subset=split_subset, task=task)
        test_dataset = CombinedCaptainCook4DDataset(
            root_dir=root_dir,
            task=task,
            features_subdir=cfg["dataset"].get("features_subdir", "I3D"),
            default_fps=cfg["dataset"].get("default_fps", 10),
            allowed_vids=allowed_vids,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg["loader"]["num_workers"],
            collate_fn=lambda batch: batch,
        )
    else:
        test_dataset = make_dataset(
            cfg["dataset_name"], False, None, cfg["test_split"], use_gcn, **cfg["dataset"]
        )
        test_loader = make_data_loader(
            test_dataset, False, None, 1, cfg["loader"]["num_workers"]
        )

    if len(test_dataset) == 0:
        print(f"[Warning] No test videos matched for task {task} (split_subset={split_subset}).")
        return None

    id2name = build_action_name_map(cfg)
    action_tokens = None
    if eval_ed:
        if clip_model is None:
            clip_model = load_clip_model(cfg["devices"][0])
        else:
            clip_model = (clip_model[0].to(cfg["devices"][0]), clip_model[1], clip_model[2])

        if cfg["dataset_name"] == "CaptainCook4D":
            with open(os.path.join(root_dir, "activity_step_collection.json"), "r") as fp:
                activity_step_collection = json.load(fp)
            with open(os.path.join(root_dir, "step_id_mapping.json"), "r") as fp:
                step_id_mapping = json.load(fp)
            action_tokens, num_classes = action_tokenize_ccp4d(
                clip_model, task, activity_step_collection, step_id_mapping, device=cfg["devices"][0]
            )
            cfg["model"]["num_classes"] = num_classes
        else:
            with open(os.path.join(root_dir, "annotation.json"), "r") as fp:
                all_annot = json.load(fp)
            action_tokens = action_tokenize_egoper(
                clip_model, task, annot=all_annot[task], device=cfg["devices"][0]
            )
    elif cfg["dataset_name"] == "CaptainCook4D":
        cfg["model"]["num_classes"] = len(id2name)

    cfg["model"]["backbone_type"] = "convGCNTransformer" if use_gcn else "convTransformer"
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=cfg["devices"])
    model = load_weights(model, ckpt_file, use_ema=use_ema)

    split_desc = f"split={os.path.basename(split_file)}[{split_subset}]" if split_file else f"split={cfg['test_split']}"
    print(f"\nStart testing ({cfg['dataset_name']} - task {task}, {split_desc}) ...")
    start = time.time()
    if eval_ed:
        results = run_full_inference(model, test_loader, clip_model, action_tokens)
    else:
        results = run_segmentation_only(model, test_loader)
    elapsed = time.time() - start

    list_of_gt_frame_action = []
    list_of_pred_frame_action = []
    for k, v in results.items():
        list_of_gt_frame_action.append(v["frame_action_label"].clone().cpu().numpy())
        list_of_pred_frame_action.append(v["frame_action_pred"].clone().cpu().numpy())

    seg_with_bg = compute_extended_seg_metrics(list_of_gt_frame_action, list_of_pred_frame_action, ignore_bg=False)
    seg_no_bg = compute_extended_seg_metrics(list_of_gt_frame_action, list_of_pred_frame_action, ignore_bg=True)
    ed_metrics = compute_ed_metrics(results) if eval_ed else {"precision": 0.0, "auc": 0.0, "eda": 0.0}

    print(
        "  [With BG]    Acc: {:5.2f}% | mIoU: {:5.2f}% | Edit: {:5.2f}% | "
        "F1@(10/25/50/75): {:5.2f}/{:5.2f}/{:5.2f}/{:5.2f} | "
        "#Pred/#GT: {:3d}/{:3d} ({:.2f}x) | Videos: {} | {:.2f}s".format(
            seg_with_bg["Accuracy"],
            seg_with_bg["mIoU"],
            seg_with_bg["Edit"],
            seg_with_bg["F1@10"],
            seg_with_bg["F1@25"],
            seg_with_bg["F1@50"],
            seg_with_bg["F1@75"],
            seg_with_bg["pred_segs"],
            seg_with_bg["gt_segs"],
            seg_with_bg["frag_ratio"],
            len(results),
            elapsed,
        )
    )
    print(
        "  [Without BG] Acc: {:5.2f}% | mIoU: {:5.2f}% | Edit: {:5.2f}% | "
        "F1@(10/25/50/75): {:5.2f}/{:5.2f}/{:5.2f}/{:5.2f} | "
        "#Pred/#GT: {:3d}/{:3d} ({:.2f}x)".format(
            seg_no_bg["Accuracy"],
            seg_no_bg["mIoU"],
            seg_no_bg["Edit"],
            seg_no_bg["F1@10"],
            seg_no_bg["F1@25"],
            seg_no_bg["F1@50"],
            seg_no_bg["F1@75"],
            seg_no_bg["pred_segs"],
            seg_no_bg["gt_segs"],
            seg_no_bg["frag_ratio"],
        )
    )
    if eval_ed:
        print(
            "  [Mistake ED] Precision: {:5.2f}% | Micro AUC: {:5.2f}% | EDA: {:5.2f}%".format(
                ed_metrics["precision"], ed_metrics["auc"], ed_metrics["eda"]
            )
        )

    seg_outputs = dict(seg_with_bg)
    seg_outputs["no_bg"] = seg_no_bg
    seg_outputs.update(ed_metrics)

    fps = cfg["dataset"].get("default_fps", 10.0)
    if show_videos > 0:
        print_video_segments(results, id2name, fps=fps, max_videos=show_videos)

    if export_video_dir:
        exported = 0
        for vid, v_res in results.items():
            if exported >= max_export_videos:
                break
            exported += 1
            src = find_video_source(video_root, vid)
            out_mp4 = os.path.join(export_video_dir, f"task{task}_{vid}_seg.mp4")
            render_segmentation_video(
                vid=vid,
                v_res=v_res,
                id2name=id2name,
                out_path=out_mp4,
                video_source=src,
                feat_fps=fps,
                out_fps=out_fps,
            )

    if save_json:
        export_data = {
            "task": task,
            "split_file": split_file,
            "split_subset": split_subset if split_file else cfg["test_split"],
            "metrics_with_bg": {k: float(v) for k, v in seg_with_bg.items()},
            "metrics_no_bg": {k: float(v) for k, v in seg_no_bg.items()},
            "videos": {},
        }
        for vid, v in results.items():
            export_data["videos"][vid] = {
                "gt_segments_sec": (v["gt_segments"].numpy() / fps).tolist(),
                "gt_labels": v["gt_segments_labels"].numpy().tolist(),
                "pred_segments_sec": (v["segments"].numpy() / fps).tolist(),
                "pred_labels": v["segment_action_pred"].numpy().tolist(),
            }
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        with open(save_json, "w") as fp:
            json.dump(export_data, fp, indent=2)
        print(f"Saved detailed segmentation predictions to: {save_json}")

    return seg_outputs


def evaluate_all_ccp4d(
    config_path,
    ckpt_dir,
    eval_ed=True,
    use_ema=False,
    split_file=None,
    split_subset="test",
    video_root=None,
    export_video_dir=None,
    max_export_videos=1,
    out_fps=5.0,
):
    """Evaluate all available CaptainCook4D recipe checkpoints in `ckpt_dir`."""
    cfg_base = load_config(config_path)
    root_dir = _resolve_root_dir(cfg_base)
    with open(os.path.join(root_dir, "id_activity_mapping.json"), "r") as fp:
        id_activity = json.load(fp)

    clip_model = load_clip_model(cfg_base["devices"][0]) if eval_ed else None

    all_metrics = []
    print(f"Scanning CaptainCook4D checkpoints in: {ckpt_dir}")
    for task_id, activity_name in id_activity.items():
        pattern = os.path.join(ckpt_dir, f"recipe_{task_id}_*.pth.tar")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[Skip] Recipe {task_id} ({activity_name}): checkpoint not found ({pattern})")
            continue
        cfg = load_config(config_path)
        cfg["dataset"]["task"] = str(task_id)
        out = evaluate_task(
            cfg,
            matches[0],
            clip_model=clip_model,
            eval_ed=eval_ed,
            use_gcn=False,
            use_ema=use_ema,
            split_file=split_file,
            split_subset=split_subset,
            video_root=video_root,
            export_video_dir=export_video_dir,
            max_export_videos=max_export_videos,
            out_fps=out_fps,
        )
        if out is not None:
            out["task_id"] = task_id
            out["activity"] = activity_name
            all_metrics.append(out)

    if all_metrics:
        def _print_summary_table(title, items, show_ed=False):
            width = 136 if show_ed else 112
            print("\n" + "=" * width)
            print(f" {title}")
            print("=" * width)
            ed_hdr = f" {'Prec':>6} {'AUC':>6} {'EDA':>6}" if show_ed else ""
            header = (
                f"{'Task':<5} {'Recipe':<31} {'Acc':>6} {'mIoU':>6} {'Edit':>6} "
                f"{'F1@10':>6} {'F1@25':>6} {'F1@50':>6} {'F1@75':>6} "
                f"{'#Pred/#GT':>11} {'Frag':>6}{ed_hdr}"
            )
            print(header)
            print("-" * width)
            for tid, rname, m in items:
                seg_str = f"{m['pred_segs']}/{m['gt_segs']}"
                ed_row = f" {m['precision']:6.2f} {m['auc']:6.2f} {m['eda']:6.2f}" if show_ed else ""
                print(
                    f"{tid:<5} {rname[:31]:<31} "
                    f"{m['Accuracy']:6.2f} {m['mIoU']:6.2f} {m['Edit']:6.2f} "
                    f"{m['F1@10']:6.2f} {m['F1@25']:6.2f} {m['F1@50']:6.2f} {m['F1@75']:6.2f} "
                    f"{seg_str:>11} {m['frag_ratio']:5.2f}x{ed_row}"
                )
            print("-" * width)
            mean_acc = np.mean([m["Accuracy"] for _, _, m in items])
            mean_iou = np.mean([m["mIoU"] for _, _, m in items])
            mean_edit = np.mean([m["Edit"] for _, _, m in items])
            mean_f1_10 = np.mean([m["F1@10"] for _, _, m in items])
            mean_f1_25 = np.mean([m["F1@25"] for _, _, m in items])
            mean_f1_50 = np.mean([m["F1@50"] for _, _, m in items])
            mean_f1_75 = np.mean([m["F1@75"] for _, _, m in items])
            tot_pred = sum(m["pred_segs"] for _, _, m in items)
            tot_gt = sum(m["gt_segs"] for _, _, m in items)
            tot_frag = (tot_pred / tot_gt) if tot_gt > 0 else 0.0
            tot_seg_str = f"{tot_pred}/{tot_gt}"
            if show_ed:
                mean_prec = np.mean([m["precision"] for _, _, m in items])
                mean_auc = np.mean([m["auc"] for _, _, m in items])
                mean_eda = np.mean([m["eda"] for _, _, m in items])
                ed_avg = f" {mean_prec:6.2f} {mean_auc:6.2f} {mean_eda:6.2f}"
            else:
                ed_avg = ""
            print(
                f"{'AVG':<5} {'(' + str(len(items)) + ' recipes)':<31} "
                f"{mean_acc:6.2f} {mean_iou:6.2f} {mean_edit:6.2f} "
                f"{mean_f1_10:6.2f} {mean_f1_25:6.2f} {mean_f1_50:6.2f} {mean_f1_75:6.2f} "
                f"{tot_seg_str:>11} {tot_frag:5.2f}x{ed_avg}"
            )
            if show_ed:
                print(
                    f"{'PAPER':<5} {'(Table 2 - CaptainCook4D)':<31} "
                    f"{'--':>6} {'--':>6} {'--':>6} "
                    f"{'--':>6} {'--':>6} {'--':>6} {'--':>6} "
                    f"{'--':>11} {'--':>6} {68.10:6.2f} {62.50:6.2f} {71.90:6.2f}"
                )
            print("=" * width)

        with_bg_items = [(m["task_id"], m["activity"], m) for m in all_metrics]
        no_bg_items = [(m["task_id"], m["activity"], m["no_bg"]) for m in all_metrics]
        _print_summary_table(
            "SUMMARY TABLE 1: WITH BACKGROUND (Class 0 Included) + MISTAKE DETECTION (Prec / AUC / EDA)",
            with_bg_items,
            show_ed=eval_ed,
        )
        _print_summary_table("SUMMARY TABLE 2: WITHOUT BACKGROUND (Action Steps Only, Class > 0)", no_bg_items, show_ed=False)


def main():
    parser = argparse.ArgumentParser(
        description="Test Action Segmentation AND Mistake Detection (Precision/AUC/EDA), with MP4 export."
    )
    parser.add_argument("config", type=str, help="path to config file (e.g. configs/ccp4d.yaml)")
    parser.add_argument(
        "ckpt",
        type=str,
        help="path to a checkpoint file (.pth.tar) or checkpoint directory",
    )
    parser.add_argument("--task", type=str, default=None, help="override dataset.task (e.g. '1')")
    parser.add_argument(
        "--all_recipes",
        action="store_true",
        help="evaluate all 24 CaptainCook4D recipes found in the ckpt folder",
    )
    parser.add_argument(
        "--seg_only",
        action="store_true",
        help="skip OpenCLIP and evaluate only Action Segmentation (faster)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=DEFAULT_SPLIT_FILE,
        help=(
            "path to CaptainCook4D split json (default: tests/person_data_split_combined.json). "
            "Pass 'none' to use the default AEM error-only test split."
        ),
    )
    parser.add_argument(
        "--split_subset",
        type=str,
        default="test",
        help="which split key(s) to evaluate from --split_file: 'test', 'val', 'train', or 'val,test' (default: 'test')",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=None,
        help="optional directory containing raw CaptainCook4D videos (<vid>_360p.mp4) or 1fps frame folders",
    )
    parser.add_argument(
        "--export_video_dir",
        type=str,
        default=None,
        help="directory to export visualization .mp4 videos comparing GT and Predicted segments",
    )
    parser.add_argument(
        "--max_export_videos",
        type=int,
        default=2,
        help="max number of videos to render per recipe when --export_video_dir is set (default: 2)",
    )
    parser.add_argument(
        "--out_fps",
        type=float,
        default=5.0,
        help="frame rate of exported visualization MP4 video (default: 5.0)",
    )
    parser.add_argument("--is_best", action="store_true")
    parser.add_argument("-epoch", type=int, default=-1)
    parser.add_argument("-t", "--topk", default=-1, type=int)
    parser.add_argument("--use_gcn", default=False, action="store_true")
    parser.add_argument("--use_ema", default=False, action="store_true")
    parser.add_argument(
        "--show_videos",
        type=int,
        default=2,
        help="number of videos to print detailed GT vs Predicted segments (default: 2)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="optional path to save predicted vs GT segments as JSON",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")

    split_file = None if (args.split_file and args.split_file.lower() == "none") else args.split_file
    eval_ed = not args.seg_only

    if args.all_recipes:
        evaluate_all_ccp4d(
            args.config,
            args.ckpt,
            eval_ed=eval_ed,
            use_ema=args.use_ema,
            split_file=split_file,
            split_subset=args.split_subset,
            video_root=args.video_root,
            export_video_dir=args.export_video_dir,
            max_export_videos=args.max_export_videos,
            out_fps=args.out_fps,
        )
        return

    cfg = load_config(args.config)
    if args.task is not None:
        cfg["dataset"]["task"] = args.task
    if args.topk > 0:
        cfg["model"]["test_cfg"]["max_seg_num"] = args.topk

    ckpt_file = resolve_ckpt_file(args.ckpt, args.epoch, args.is_best)
    clip_model = load_clip_model(cfg["devices"][0]) if eval_ed else None
    evaluate_task(
        cfg,
        ckpt_file,
        clip_model=clip_model,
        eval_ed=eval_ed,
        use_gcn=args.use_gcn,
        use_ema=args.use_ema,
        show_videos=args.show_videos,
        save_json=args.save_json,
        split_file=split_file,
        split_subset=args.split_subset,
        video_root=args.video_root,
        export_video_dir=args.export_video_dir,
        max_export_videos=args.max_export_videos,
        out_fps=args.out_fps,
    )


if __name__ == "__main__":
    main()
