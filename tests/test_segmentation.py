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
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from metric import evaluate_seg

warnings.filterwarnings("ignore")

DEFAULT_SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person_data_split_combined.json")

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


def build_action_name_map(cfg):
    """Return {action_id: action_name} for human-readable segment inspection."""
    root_dir = cfg["dataset"]["root_dir"]
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
    """Evaluate segmentation branch on one task/recipe without loading OpenCLIP."""
    _ = fix_random_seed(0, include_cuda=True)

    task = str(cfg["dataset"]["task"])
    if split_file and cfg["dataset_name"] == "CaptainCook4D":
        allowed_vids = load_split_video_ids(split_file, subset=split_subset, task=task)
        test_dataset = CombinedCaptainCook4DDataset(
            root_dir=cfg["dataset"]["root_dir"],
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
    if cfg["dataset_name"] == "CaptainCook4D":
        cfg["model"]["num_classes"] = len(id2name)

    cfg["model"]["backbone_type"] = "convGCNTransformer" if use_gcn else "convTransformer"
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=cfg["devices"])
    model = load_weights(model, ckpt_file, use_ema=use_ema)

    split_desc = f"split={os.path.basename(split_file)}[{split_subset}]" if split_file else f"split={cfg['test_split']}"
    print(f"\nStart testing Action Segmentation ({cfg['dataset_name']} - task {task}, {split_desc}) ...")
    start = time.time()
    results = run_segmentation_only(model, test_loader)
    elapsed = time.time() - start

    list_of_gt_frame_action = []
    list_of_pred_frame_action = []
    for k, v in results.items():
        list_of_gt_frame_action.append(v["frame_action_label"].clone().cpu().numpy())
        list_of_pred_frame_action.append(v["frame_action_pred"].clone().cpu().numpy())

    seg_outputs = evaluate_seg(list_of_gt_frame_action, list_of_pred_frame_action)
    print(
        "Segmentation Accuracy: {:.2f}% | mIoU: {:.2f}% | Edit: {:.2f}% | F1@0.5: {:.2f}% | Videos: {} | Time: {:.2f}s".format(
            seg_outputs["Accuracy"],
            seg_outputs["mIoU"],
            seg_outputs["Edit"],
            seg_outputs["F1@0.5"],
            len(results),
            elapsed,
        )
    )

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
            "metrics": {k: float(v) for k, v in seg_outputs.items()},
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
    root_dir = cfg_base["dataset"]["root_dir"]
    with open(os.path.join(root_dir, "id_activity_mapping.json"), "r") as fp:
        id_activity = json.load(fp)

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
        print("\n" + "=" * 75)
        print(f"{'Task':<6} {'Recipe':<34} {'Acc':>7} {'mIoU':>7} {'Edit':>7} {'F1@0.5':>7}")
        print("-" * 75)
        for m in all_metrics:
            print(
                f"{m['task_id']:<6} {m['activity']:<34} "
                f"{m['Accuracy']:7.2f} {m['mIoU']:7.2f} {m['Edit']:7.2f} {m['F1@0.5']:7.2f}"
            )
        print("-" * 75)
        mean_acc = np.mean([m["Accuracy"] for m in all_metrics])
        mean_iou = np.mean([m["mIoU"] for m in all_metrics])
        mean_edit = np.mean([m["Edit"] for m in all_metrics])
        mean_f1 = np.mean([m["F1@0.5"] for m in all_metrics])
        print(
            f"{'AVG':<6} {'(' + str(len(all_metrics)) + ' recipes)':<34} "
            f"{mean_acc:7.2f} {mean_iou:7.2f} {mean_edit:7.2f} {mean_f1:7.2f}"
        )
        print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Test ONLY the Action Segmentation branch (supports person_data_split_combined.json & MP4 export)."
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

    if args.all_recipes:
        evaluate_all_ccp4d(
            args.config,
            args.ckpt,
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
    evaluate_task(
        cfg,
        ckpt_file,
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
