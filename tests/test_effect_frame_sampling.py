"""Test & Inspect the Effect Frame Sampling Algorithm (AEM - ICLR 2026).

This script runs the exact Effect Frame Sampling algorithm proposed in Section 3.3
of the paper (and implemented in `data_preparation/captaincook4d/select_effect_frames.py`):

  1. Sample video frames at 1 fps (from either a raw .mp4 video or a folder of 1 fps frames).
  2. Compute global sharpness distribution across the video using Laplacian variance:
       v_t = Var(Laplacian(gray(f_t)))
       q_t = clip((v_t - P_5) / (P_95 - P_5), 0, 1)
  3. For each action segment [ceil(s), floor(e)] (merging trailing BG as in paper code):
     - Compute raw semantic alignment score via EVA02-CLIP-L-14-336:
         c_t = max_{p in Prompts} cos(Phi_vis(f_t), Psi_txt(p))
     - Min-max normalize both scores WITHIN the segment to [0, 1]:
         C_t = (c_t - min(c)) / (max(c) - min(c))
         Q_t = (q_t - min(q)) / (max(q) - min(q))
     - Combine with weighting factor alpha (default 0.7):
         F_t = alpha * C_t + (1 - alpha) * Q_t
     - Select the frame t* = argmax_t F_t as the Effect Frame.

It prints a detailed frame-by-frame score table for each segment so you can see
WHY a specific frame won (vs. other frames and vs. the naive Last Frame baseline),
compares against the paper's pre-extracted `effect_frames/` if available, and
optionally saves side-by-side comparison images to `--output_dir`.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
DATA_PREP_DIR = os.path.join(REPO_ROOT, "data_preparation")
if DATA_PREP_DIR not in sys.path:
    sys.path.insert(0, DATA_PREP_DIR)

from common import (
    ccp4d_remapped_idx2action,
    clarity_score,
    clip_content_score,
    effect_prompt,
    laplacian_var,
    load_eva_clip,
    read_json,
)


def find_video_source(video_root, vid):
    """Find either a 1fps frame directory or a raw video file (.mp4/.avi/.mkv) for `vid`."""
    if not video_root or not os.path.exists(video_root):
        return None, None

    # 1. Check direct frame directory candidates
    dir_candidates = [
        os.path.join(video_root, f"{vid}_360p"),
        os.path.join(video_root, vid),
        os.path.join(video_root, "frames", f"{vid}_360p"),
        os.path.join(video_root, "frames", vid),
    ]
    for d in dir_candidates:
        if os.path.isdir(d):
            pngs = list(Path(d).glob("*.png")) + list(Path(d).glob("*.jpg"))
            if pngs:
                return "dir", d

    # 2. Check direct video file candidates
    for ext in (".mp4", ".MP4", ".avi", ".mkv", ".mov"):
        for name in (f"{vid}_360p{ext}", f"{vid}{ext}"):
            p = os.path.join(video_root, name)
            if os.path.isfile(p):
                return "video", p

    # 3. Recursive walk fallback
    for root, dirs, files in os.walk(video_root):
        for d in dirs:
            if d in (vid, f"{vid}_360p"):
                full_d = os.path.join(root, d)
                pngs = list(Path(full_d).glob("*.png")) + list(Path(full_d).glob("*.jpg"))
                if pngs:
                    return "dir", full_d
        for f in files:
            if f.startswith(vid + ".") or f.startswith(vid + "_"):
                if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                    return "video", os.path.join(root, f)

    return None, None


def load_1fps_frames(src_type, src_path):
    """Load 1-fps BGR frames as a 1-indexed dict {frame_sec_idx: img_bgr}.

    Matches `select_effect_frames.py` where `000001.png` corresponds to second 1.
    If reading from an `.mp4` video, samples at 1 fps (t = 1, 2, 3, ... seconds).
    """
    import cv2

    frames_dict = {}
    if src_type == "dir":
        files = sorted(list(Path(src_path).glob("*.png")) + list(Path(src_path).glob("*.jpg")))
        for idx, p in enumerate(files, start=1):
            stem = p.stem
            frame_num = int(stem) if stem.isdigit() else idx
            img = cv2.imread(str(p))
            if img is not None:
                frames_dict[frame_num] = img
    elif src_type == "video":
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {src_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps)
        print(f"  [Video Info] {os.path.basename(src_path)} | FPS: {fps:.2f} | Total frames: {total_frames} (~{duration_sec}s)")
        # Read sequentially for speed and accuracy
        sec = 1
        target_frame_idx = int(round((sec - 1) * fps))
        cur_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx == target_frame_idx:
                frames_dict[sec] = frame
                sec += 1
                target_frame_idx = int(round((sec - 1) * fps))
            cur_idx += 1
        cap.release()
    return frames_dict


def build_effect_descriptions(data_root, recipe, vid, action_list, gpt_api_key=None):
    """Obtain effect description sentences for each action in `recipe`.

    Priority:
      1. `<data_root>/effect_desc_ccp4d.json` if it exists.
      2. If `--api_key` or `OPENAI_API_KEY` is provided, query `gpt-4o-mini` with the
         exact 2-shot prompt from `gen_effect_desc.py`.
      3. Automatic zero-API-key fallback constructed from the repo's existing GPT-4o
         annotations (`gpt4o_action_objects.json` + `openset_detection_ccp4d.json`),
         generating 3 concise post-condition sentences describing the resulting object states.
    """
    effect_desc_path = os.path.join(data_root, "effect_desc_ccp4d.json")
    if os.path.isfile(effect_desc_path):
        all_desc = read_json(effect_desc_path)
        if recipe in all_desc:
            print(f"  [Prompts] Loaded official effect_desc_ccp4d.json for recipe '{recipe}'")
            return all_desc[recipe], "effect_desc_ccp4d.json"

    # Option 2: Live GPT-4o-mini generation if key is available
    api_key = gpt_api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        from data_preparation.captaincook4d.gen_effect_desc import build_prompt, parse_sentences
        from common import get_openai_client

        client = get_openai_client(api_key)
        recipe_desc = {}
        for action in set(action_list):
            if action == "BG":
                continue
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": build_prompt(recipe, action)}],
            )
            recipe_desc[action] = parse_sentences(resp.choices[0].message.content)
        return recipe_desc, "gpt-4o-mini (live)"

    # Option 3: Construct from repo's existing GPT-4o annotations
    obj_path = os.path.join(data_root, "gpt4o_action_objects.json")
    det_path = os.path.join(data_root, "detection", "openset_detection_ccp4d.json")
    sg_path = os.path.join(data_root, "scene_graph_json", "scene_graph.json")

    action_objs = read_json(obj_path).get(recipe, {}) if os.path.isfile(obj_path) else {}
    det_data = read_json(det_path) if os.path.isfile(det_path) else {}
    sg_data = read_json(sg_path) if os.path.isfile(sg_path) else {}

    # Find video-specific or recipe-wide descriptions in openset_detection / scene_graph
    vid_det_map = {}
    vid_sg_map = {}
    for task_id, task_vids in det_data.items():
        if vid in task_vids:
            for seg in task_vids[vid]:
                if seg.get("action") and seg.get("action") != "BG":
                    vid_det_map[seg["action"]] = seg
    for task_id, task_vids in sg_data.items():
        if vid in task_vids:
            for seg in task_vids[vid]:
                if seg.get("action") and seg.get("action") != "BG":
                    vid_sg_map[seg["action"]] = seg

    recipe_desc = {}
    for action in set(action_list):
        if action == "BG":
            continue
        sentences = []
        # Check scene graph relation/attribute sentences first
        if action in vid_sg_map:
            sg_seg = vid_sg_map[action]
            sentences.extend(sg_seg.get("relation_sentence", []))
            sentences.extend(sg_seg.get("attribute_sentence", []))
        # Check openset_detection descriptions (Location Relationship, Location of OBJECT A/B)
        if len(sentences) < 3 and action in vid_det_map:
            det_seg = vid_det_map[action]
            for key in ("Location Relationship", "Location of OBJECT A", "Description of OBJECT A"):
                val = det_seg.get(key)
                if val and val not in sentences:
                    sentences.append(val)
        # Fallback to objectA / objectB templates matching gen_effect_desc.py style
        if not sentences:
            objs = action_objs.get(action, {})
            obj_a = objs.get("objectA", "the ingredient")
            obj_b = objs.get("objectB", "the container")
            sentences = [
                f"The {obj_a} is with the {obj_b}",
                f"The {obj_b} contains or holds the {obj_a}",
                f"The step {action} is completed with {obj_a} and {obj_b}",
            ]
        recipe_desc[action] = sentences[:3]

    return recipe_desc, "repo GPT-4o annotations (openset_detection + scene_graph)"


def parse_existing_effect_frame(effect_frames_root, vid, start_time, end_time):
    """Look up the paper's pre-extracted effect frame in `data/captaincook4d/effect_frames/<vid>`."""
    vid_dir = os.path.join(effect_frames_root, vid)
    if not os.path.isdir(vid_dir):
        return None
    target_prefix = f"{start_time}_"
    candidates = []
    for sub in sorted(os.listdir(vid_dir)):
        if sub == f"{start_time}_{end_time}" or sub.startswith(target_prefix):
            sub_dir = os.path.join(vid_dir, sub)
            if os.path.isdir(sub_dir):
                jpgs = sorted([f for f in os.listdir(sub_dir) if f.endswith(".jpg") or f.endswith(".png")])
                if jpgs:
                    candidates.append((sub, os.path.join(sub_dir, jpgs[0]), jpgs[0]))
    if not candidates:
        return None
    # Prefer exact start_end match
    for sub, full_p, fname in candidates:
        if sub == f"{start_time}_{end_time}":
            return _parse_fname_scores(sub, full_p, fname)
    return _parse_fname_scores(*candidates[0])


def _parse_fname_scores(sub, full_p, fname):
    """Parse `<frame>_<F>_<C>_<Q>.jpg` e.g. `000037_0.87_1.00_0.58.jpg`."""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) >= 4:
        try:
            return {
                "folder": sub,
                "path": full_p,
                "filename": fname,
                "frame_idx": int(parts[0]),
                "F": float(parts[1]),
                "C": float(parts[2]),
                "Q": float(parts[3]),
            }
        except ValueError:
            pass
    return {"folder": sub, "path": full_p, "filename": fname, "frame_idx": None, "F": None, "C": None, "Q": None}


def save_segment_comparison_image(seg_idx, vid, action, start_time, end_time,
                                  scored_rows, paper_ref, output_dir):
    """Save a visual comparison strip: Top-1 Effect Frame vs Top-2 vs Most Blurry vs Last Frame."""
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    # Sort by F descending for top frames, and by chronological order for Last Frame
    by_f = sorted(scored_rows, key=lambda x: x["F"], reverse=True)
    by_time = sorted(scored_rows, key=lambda x: x["frame_idx"])
    by_blur = sorted(scored_rows, key=lambda x: x["Q"])

    panels_info = [
        ("TOP-1 EFFECT FRAME (Selected)", by_f[0], (40, 200, 80)),
    ]
    if len(by_f) > 1:
        panels_info.append(("TOP-2 CANDIDATE", by_f[1], (200, 180, 50)))
    if len(by_blur) > 0:
        panels_info.append(("LOWEST CLARITY (Blurry)", by_blur[0], (60, 120, 240)))
    panels_info.append(("LAST FRAME OF SEGMENT", by_time[-1], (80, 80, 220)))

    # If paper's pre-extracted image exists on disk, add it too
    if paper_ref and paper_ref.get("path") and os.path.isfile(paper_ref["path"]):
        ref_img = cv2.imread(paper_ref["path"])
        if ref_img is not None:
            panels_info.append((
                f"PAPER PRE-EXTRACTED (#{paper_ref['frame_idx']})",
                {
                    "frame_idx": paper_ref["frame_idx"],
                    "F": paper_ref["F"] or 0.0,
                    "C": paper_ref["C"] or 0.0,
                    "Q": paper_ref["Q"] or 0.0,
                    "c_raw": 0.0,
                    "v_raw": 0.0,
                    "img": ref_img,
                },
                (180, 105, 255),
            ))

    target_w, target_h = 380, 240
    header_h = 85
    cards = []
    for title, item, color in panels_info:
        img_resized = cv2.resize(item["img"], (target_w, target_h))
        card = np.zeros((target_h + header_h, target_w, 3), dtype=np.uint8)
        card[:header_h, :] = (28, 28, 34)
        cv2.rectangle(card, (0, 0), (target_w - 1, header_h - 1), color, 2)
        card[header_h:, :] = img_resized

        cv2.putText(card, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        cv2.putText(
            card,
            f"Frame #{item['frame_idx']:04d} ({item['frame_idx']}s) | Final F = {item['F']:.3f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            card,
            f"Semantic C = {item['C']:.2f} | Clarity Q = {item['Q']:.2f}",
            (10, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (190, 210, 250),
            1,
            cv2.LINE_AA,
        )
        cards.append(card)

    strip = np.hstack(cards)
    # Add top banner with action name
    banner_h = 46
    canvas = np.zeros((strip.shape[0] + banner_h, strip.shape[1], 3), dtype=np.uint8)
    canvas[:banner_h, :] = (18, 18, 22)
    canvas[banner_h:, :] = strip
    banner_text = f"[{vid}] Seg #{seg_idx:02d} ({start_time}s - {end_time}s): {action}"
    cv2.putText(canvas, banner_text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    out_file = os.path.join(output_dir, f"{vid}_seg{seg_idx:02d}_{start_time}_{end_time}.jpg")
    cv2.imwrite(out_file, canvas)
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Run and inspect the paper's Effect Frame Sampling algorithm (CLIP Semantic + Laplacian Clarity)."
    )
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data", "captaincook4d"),
                        help="Path to data/captaincook4d")
    parser.add_argument("--video_root", default="/content/drive/MyDrive/AEM/datasets/CaptainCook4D",
                        help="Folder containing raw .mp4 videos or <vid>_360p 1fps frames")
    parser.add_argument("--task", type=str, default="1",
                        help="Recipe/task ID in CaptainCook4D (e.g. '1' for Microwave Egg Sandwich)")
    parser.add_argument("--video_id", type=str, default="1_7",
                        help="Video ID to inspect (e.g. '1_7', '1_14', '1_19')")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weighting factor alpha in F = alpha * C + (1 - alpha) * Q (paper default: 0.7)")
    parser.add_argument("--max_segments", type=int, default=5,
                        help="Max number of action segments to print in full detail (0 = all)")
    parser.add_argument("--show_all_frames", action="store_true",
                        help="Print every single second/frame in each segment (by default prints Top-5, Bottom-2, and Last Frame if segment > 15s)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory to save side-by-side comparison images (.jpg)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Optional OpenAI API key if you want to generate prompts live with gpt-4o-mini")
    args = parser.parse_args()

    import cv2
    import numpy as np

    # 1. Load annotations for the specified task & video
    non_err_path = os.path.join(args.data_root, "non_error_samples_processed.json")
    err_path = os.path.join(args.data_root, "error_samples_processed.json")
    annot = read_json(non_err_path)
    if args.task not in annot or args.video_id not in annot.get(args.task, {}):
        err_annot = read_json(err_path) if os.path.isfile(err_path) else {}
        if args.task in err_annot and args.video_id in err_annot[args.task]:
            annot = err_annot
        else:
            avail = list(annot.get(args.task, {}).keys())[:10]
            raise SystemExit(
                f"Video '{args.video_id}' not found in task '{args.task}'. Available normal videos: {avail}"
            )

    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    activity_steps = read_json(os.path.join(args.data_root, "activity_step_collection.json"))
    step_id_map = read_json(os.path.join(args.data_root, "step_id_mapping.json"))
    idx2action = ccp4d_remapped_idx2action(id_activity, activity_steps, step_id_map)

    recipe = id_activity[args.task]
    vid_info = annot[args.task][args.video_id]

    # Exact segment preparation from `select_effect_frames.py` (lines 118-126):
    # Trailing BG segments are merged into the preceding action segment!
    action_list = [idx2action[args.task][a] if a != 0 else "BG" for a in vid_info["labels"]]
    action_bounds = [[s / 10.0, e / 10.0] for s, e in vid_info["segments"]]
    raw_bounds = [list(b) for b in action_bounds]
    for i in range(len(action_list) - 1, -1, -1):
        if action_list[i] == "BG":
            if i > 0:
                action_bounds[i - 1][1] = action_bounds[i][1]
            del action_list[i]
            del action_bounds[i]
            del raw_bounds[i]

    print("=" * 110)
    print(f" EFFECT FRAME SAMPLING INSPECTOR | Recipe [{args.task}]: {recipe} | Video: {args.video_id}")
    print(f" Formula: F_t = {args.alpha:.2f} * C_norm(EVA02-CLIP) + {1.0 - args.alpha:.2f} * Q_norm(Laplacian Clarity)")
    print("=" * 110)

    # 2. Locate video or 1fps frames
    src_type, src_path = find_video_source(args.video_root, args.video_id)
    effect_frames_root = os.path.join(args.data_root, "effect_frames")

    if src_type is None:
        print(f"[!] Could not find raw video or 1fps frames for '{args.video_id}' under: {args.video_root}")
        print(f"    Checking pre-extracted effect_frames in: {os.path.join(effect_frames_root, args.video_id)}")
        if os.path.isdir(os.path.join(effect_frames_root, args.video_id)):
            print("\nPre-extracted Effect Frames from Paper (`<frame>_<F>_<C>_<Q>.jpg`):")
            print(f"{'Seg#':<5} {'Bounds (s)':<14} {'Selected Frame':<16} {'Final F':>9} {'Semantic C':>12} {'Clarity Q':>11}  {'Action Step'}")
            print("-" * 110)
            for idx, (action, bounds) in enumerate(zip(action_list, action_bounds), start=1):
                st, et = int(math.ceil(bounds[0])), int(math.floor(bounds[1]))
                ref = parse_existing_effect_frame(effect_frames_root, args.video_id, st, et)
                if ref:
                    print(
                        f"{idx:<5d} {f'{st}-{et}s':<14} #{ref['frame_idx']:06d} ({ref['frame_idx']}s)   "
                        f"{ref['F']:>9.2f} {ref['C']:>12.2f} {ref['Q']:>11.2f}  {action}"
                    )
        raise SystemExit(
            "\nTo run the full frame-by-frame scoring on every second of the video, pass --video_root "
            "pointing to the folder containing your CaptainCook4D .mp4 videos or 1fps frames."
        )

    print(f"  [Source] Found {src_type}: {src_path}")
    frames_dict = load_1fps_frames(src_type, src_path)
    print(f"  [Frames] Loaded {len(frames_dict)} frames sampled at 1 FPS.")

    # 3. Compute video-wide Laplacian variance percentiles (p5, p95) exactly as in select_effect_frames.py
    sorted_sec = sorted(frames_dict.keys())
    vols = [laplacian_var(cv2.cvtColor(frames_dict[s], cv2.COLOR_BGR2GRAY)) for s in sorted_sec]
    p5, p95 = np.percentile(vols, [5, 95])
    print(f"  [Clarity] Global Laplacian Variance across video: P5 = {p5:.2f}, P95 = {p95:.2f}")

    # 4. Build effect descriptions & load EVA02-CLIP
    recipe_desc, prompt_source = build_effect_descriptions(
        args.data_root, recipe, args.video_id, action_list, gpt_api_key=args.api_key
    )
    print(f"  [Prompts] Source: {prompt_source}")
    print("  [CLIP]    Loading EVA02-CLIP-L-14-336 (merged2b_s6b_b61k)...")
    clip_bundle = load_eva_clip()
    model, preprocess, tokenizer, device = clip_bundle

    num_segs = len(action_list) if args.max_segments <= 0 else min(len(action_list), args.max_segments)
    print(f"\nEvaluating {num_segs}/{len(action_list)} action segments in video '{args.video_id}'...\n")

    summary_rows = []

    for seg_idx, (action, raw_b, bounds) in enumerate(
        zip(action_list[:num_segs], raw_bounds[:num_segs], action_bounds[:num_segs]), start=1
    ):
        start_time = int(math.ceil(bounds[0]))
        end_time = int(math.floor(bounds[1]))
        raw_end = int(math.floor(raw_b[1]))

        prompts = effect_prompt(recipe_desc[action])

        rows = []
        for frame_idx in range(max(start_time, 1), end_time + 1):
            img = frames_dict.get(frame_idx)
            if img is None:
                continue
            v_raw = float(laplacian_var(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            q_global = float(clarity_score(v_raw, p5, p95))
            c_raw = float(clip_content_score(model, preprocess, tokenizer, device, img, prompts))
            rows.append({
                "frame_idx": frame_idx,
                "c_raw": c_raw,
                "v_raw": v_raw,
                "q_global": q_global,
                "img": img,
            })

        if not rows:
            print(f"  [Seg #{seg_idx:02d}] No frames in range [{start_time}, {end_time}]s, skipping.")
            continue

        # Segment-level min-max normalization (exact match with lines 69-73 of select_effect_frames.py)
        c_arr = np.array([r["c_raw"] for r in rows])
        q_arr = np.array([r["q_global"] for r in rows])
        c_norm = (c_arr - c_arr.min()) / (c_arr.max() - c_arr.min() + 1e-9)
        q_norm = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min() + 1e-9)
        f_arr = args.alpha * c_norm + (1.0 - args.alpha) * q_norm

        for i, r in enumerate(rows):
            r["C"] = float(c_norm[i])
            r["Q"] = float(q_norm[i])
            r["F"] = float(f_arr[i])

        # Rank frames by Final score F
        ranked = sorted(rows, key=lambda x: x["F"], reverse=True)
        for rank_i, r in enumerate(ranked, start=1):
            r["rank"] = rank_i

        best_row = ranked[0]
        last_row = rows[-1]
        paper_ref = parse_existing_effect_frame(effect_frames_root, args.video_id, start_time, end_time)

        print("=" * 110)
        print(f"SEGMENT #{seg_idx:02d}: {action}")
        print(f"  Time Range : [{bounds[0]:.1f}s -> {bounds[1]:.1f}s] -> 1fps frames [{start_time}..{end_time}] "
              f"({len(rows)} frames; action end before BG merge: {raw_end}s)")
        print("  CLIP Prompts:")
        for p_i, p_str in enumerate(prompts, start=1):
            print(f"    ({p_i}) \"{p_str}\"")
        print("-" * 110)
        print(f"  {'Frame (sec)':<13} | {'Raw CLIP':>9} | {'Lapl.Var':>9} | {'Clar(P5-95)':>11} | "
              f"{'Norm C':>8} | {'Norm Q':>8} | {'Final F':>9} | {'Rank':>5} | {'Note'}")
        print("-" * 110)

        # Decide which rows to display so terminal output is clean and informative
        if args.show_all_frames or len(rows) <= 18:
            display_rows = rows
        else:
            top_set = {r["frame_idx"] for r in ranked[:5]}
            bot_set = {r["frame_idx"] for r in ranked[-2:]}
            special_set = {rows[0]["frame_idx"], last_row["frame_idx"], min(raw_end, last_row["frame_idx"])}
            if paper_ref and paper_ref.get("frame_idx"):
                special_set.add(paper_ref["frame_idx"])
            keep_indices = top_set | bot_set | special_set
            display_rows = [r for r in rows if r["frame_idx"] in keep_indices]

        prev_fidx = None
        for r in display_rows:
            if prev_fidx is not None and r["frame_idx"] - prev_fidx > 1:
                print(f"  {'...':^13} | {'...':^9} | {'...':^9} | {'...':^11} | {'...':^8} | {'...':^8} | {'...':^9} | {'...':^5} |")
            notes = []
            if r["frame_idx"] == best_row["frame_idx"]:
                notes.append("<-- SELECTED TOP-1 (Effect Frame)")
            elif r["rank"] <= 3:
                notes.append(f"Top-{r['rank']}")
            if r["frame_idx"] == last_row["frame_idx"]:
                notes.append("Last Frame (w/ BG)")
            if r["frame_idx"] == raw_end and raw_end != last_row["frame_idx"]:
                notes.append("Action End (no BG)")
            if r["Q"] == 0.0:
                notes.append("Most Blurry")
            if paper_ref and r["frame_idx"] == paper_ref.get("frame_idx"):
                notes.append(f"[Paper GT: #{paper_ref['frame_idx']} F={paper_ref['F']:.2f}]")

            print(
                f"  #{r['frame_idx']:06d} ({r['frame_idx']:>3d}s) | "
                f"{r['c_raw']:>9.4f} | {r['v_raw']:>9.1f} | {r['q_global']:>11.3f} | "
                f"{r['C']:>8.3f} | {r['Q']:>8.3f} | {r['F']:>9.3f} | "
                f"#{r['rank']:<4d} | {'  '.join(notes)}"
            )
            prev_fidx = r["frame_idx"]

        print("-" * 110)
        print(
            f"  => SELECTED EFFECT FRAME : #{best_row['frame_idx']:06d} ({best_row['frame_idx']}s) "
            f"| Final F = {best_row['F']:.2f} (Semantic C = {best_row['C']:.2f}, Clarity Q = {best_row['Q']:.2f})"
        )
        print(
            f"  => NAIVE LAST FRAME      : #{last_row['frame_idx']:06d} ({last_row['frame_idx']}s) "
            f"| Final F = {last_row['F']:.2f} (Semantic C = {last_row['C']:.2f}, Clarity Q = {last_row['Q']:.2f}, Rank #{last_row['rank']}/{len(rows)})"
        )
        if paper_ref:
            print(
                f"  => PAPER PRE-EXTRACTED   : #{paper_ref['frame_idx']:06d} ({paper_ref['frame_idx']}s) "
                f"| File: {paper_ref['folder']}/{paper_ref['filename']} "
                f"(F = {paper_ref['F']:.2f}, C = {paper_ref['C']:.2f}, Q = {paper_ref['Q']:.2f})"
            )

        if args.output_dir:
            saved_img = save_segment_comparison_image(
                seg_idx, args.video_id, action, start_time, end_time, rows, paper_ref, args.output_dir
            )
            print(f"  => Saved comparison image: {saved_img}")
        print()

        summary_rows.append({
            "seg_idx": seg_idx,
            "action": action,
            "range": f"{start_time}-{end_time}s",
            "our_frame": best_row["frame_idx"],
            "our_scores": f"F={best_row['F']:.2f} (C={best_row['C']:.2f}, Q={best_row['Q']:.2f})",
            "last_frame": last_row["frame_idx"],
            "last_scores": f"F={last_row['F']:.2f} (C={last_row['C']:.2f}, Q={last_row['Q']:.2f})",
            "paper_frame": paper_ref["frame_idx"] if paper_ref else None,
            "paper_scores": (
                f"F={paper_ref['F']:.2f} (C={paper_ref['C']:.2f}, Q={paper_ref['Q']:.2f})"
                if paper_ref and paper_ref.get("F") is not None else "N/A"
            ),
        })

    # Print final compact summary table
    if summary_rows:
        print("=" * 115)
        print(f" SUMMARY OF SELECTED EFFECT FRAMES ({args.video_id} - {recipe})")
        print("=" * 115)
        print(f"{'Seg':<4} {'Range':<10} {'Selected (Ours)':<32} {'Last Frame Baseline':<32} {'Paper Pre-Extracted':<32}")
        print("-" * 115)
        for s in summary_rows:
            our_str = f"#{s['our_frame']:04d}s | {s['our_scores']}"
            last_str = f"#{s['last_frame']:04d}s | {s['last_scores']}"
            pap_str = f"#{s['paper_frame']:04d}s | {s['paper_scores']}" if s["paper_frame"] else "N/A"
            print(f"#{s['seg_idx']:02d}  {s['range']:<10} {our_str:<32} {last_str:<32} {pap_str:<32}")
        print("=" * 115)


if __name__ == "__main__":
    main()
