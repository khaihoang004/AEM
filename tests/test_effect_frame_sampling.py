"""Test & Audit the Effect Frame Sampling Algorithm across Single or ALL Videos (AEM - ICLR 2026).

This script supports TWO complementary modes:
  1. `--audit_paper` (Fast Audit across ALL videos / recipes, no raw .mp4 required):
     Scans all videos in `data/captaincook4d/detection/openset_detection_ccp4d.json` (and
     `data/captaincook4d/effect_frames/`) to inspect every single selected Effect Frame,
     its exact scores (Final F, Semantic C, Clarity Q), where it falls within the segment
     (`rel_pos` in [0%, 100%] of the action), and automatically flags suspicious selections:
       - [EARLY <30%]   : Frame picked near the very start of the action (before effect happens!).
       - [IN BG MERGE]  : Frame picked in the trailing Background segment after action ended.
       - [LOW CLARITY]  : Frame picked solely because C=1.00 despite blurry image (Q < 0.35).
       - [LOW SEMANTIC] : Frame picked mainly for sharpness despite poor semantic match (C < 0.70).

  2. Full Frame-by-Frame Scoring Mode (on one video, `--all_videos_in_task`, or `--all_videos`):
     Loads 1-fps frames from `--video_root`, computes EVA02-CLIP cosine similarity (`c_raw`)
     and Laplacian variance (`v_raw`) for every second in each segment, applies the paper's
     exact min-max normalization within the segment:
         C_t = (c_t - min(c)) / (max(c) - min(c))
         Q_t = (q_t - min(q)) / (max(q) - min(q))
         F_t = alpha * C_t + (1 - alpha) * Q_t
     prints the frame-by-frame score table, and exports visual comparison strips (`.jpg`)
     or contact sheets so you can see with your own eyes why each frame won.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
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

    for ext in (".mp4", ".MP4", ".avi", ".mkv", ".mov"):
        for name in (f"{vid}_360p{ext}", f"{vid}{ext}"):
            p = os.path.join(video_root, name)
            if os.path.isfile(p):
                return "video", p

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
    """Load 1-fps BGR frames as a 1-indexed dict {frame_sec_idx: img_bgr}."""
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
    """Obtain effect description sentences for each action in `recipe`."""
    effect_desc_path = os.path.join(data_root, "effect_desc_ccp4d.json")
    if os.path.isfile(effect_desc_path):
        all_desc = read_json(effect_desc_path)
        if recipe in all_desc:
            return all_desc[recipe], "effect_desc_ccp4d.json"

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

    obj_path = os.path.join(data_root, "gpt4o_action_objects.json")
    det_path = os.path.join(data_root, "detection", "openset_detection_ccp4d.json")
    sg_path = os.path.join(data_root, "scene_graph_json", "scene_graph.json")

    action_objs = read_json(obj_path).get(recipe, {}) if os.path.isfile(obj_path) else {}
    det_data = read_json(det_path) if os.path.isfile(det_path) else {}
    sg_data = read_json(sg_path) if os.path.isfile(sg_path) else {}

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
        if action in vid_sg_map:
            sg_seg = vid_sg_map[action]
            sentences.extend(sg_seg.get("relation_sentence", []))
            sentences.extend(sg_seg.get("attribute_sentence", []))
        if len(sentences) < 3 and action in vid_det_map:
            det_seg = vid_det_map[action]
            for key in ("Location Relationship", "Location of OBJECT A", "Description of OBJECT A"):
                val = det_seg.get(key)
                if val and val not in sentences:
                    sentences.append(val)
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


def diagnose_selection_flags(frame_idx, act_start, act_end, merged_end, C, Q):
    """Return list of diagnostic tags explaining why a selected frame might NOT reflect the action effect."""
    flags = []
    act_dur = max(1.0, float(act_end - act_start))
    rel_pos = (frame_idx - act_start) / act_dur  # can be > 1.0 if inside merged trailing BG

    if frame_idx <= act_start + 1 and act_dur >= 5.0:
        flags.append("AT_VERY_START(<=1s)")
    elif rel_pos < 0.30 and act_dur >= 6.0:
        flags.append(f"TOO_EARLY({rel_pos*100:.0f}%)")

    if frame_idx > act_end:
        flags.append(f"IN_TRAILING_BG(+{frame_idx - int(act_end)}s)")

    if Q is not None and Q < 0.35:
        flags.append(f"BLURRY(Q={Q:.2f})")
    if C is not None and C < 0.75:
        flags.append(f"WEAK_CLIP(C={C:.2f})")
    return flags, rel_pos


def run_paper_audit(data_root, task_filter="all", video_filter="all", max_videos_per_task=0, output_dir=None):
    """Audit all pre-extracted Effect Frames from `openset_detection_ccp4d.json` across the dataset.

    Prints per-video tables and a global statistical breakdown of why certain frames fail to
    capture the true post-action effect, plus optionally saves visual contact sheets of
    the paper's pre-extracted `effect_frames/` with their GroundingDINO bounding boxes & scores.
    """
    import statistics

    det_path = os.path.join(data_root, "detection", "openset_detection_ccp4d.json")
    id_activity = read_json(os.path.join(data_root, "id_activity_mapping.json"))
    det_data = read_json(det_path)

    wanted_tasks = None if task_filter.lower() == "all" else {t.strip() for t in task_filter.split(",") if t.strip()}
    wanted_vids = None if video_filter.lower() == "all" else {v.strip() for v in video_filter.split(",") if v.strip()}

    total_segs = 0
    early_30_count = 0
    very_start_count = 0
    in_bg_count = 0
    low_q_count = 0
    low_c_count = 0
    c_is_1_count = 0
    q_is_1_count = 0
    rel_positions = []
    suspicious_examples = []

    print("=" * 135)
    print(" GLOBAL AUDIT OF PAPER'S SELECTED EFFECT FRAMES (openset_detection_ccp4d.json)")
    print("=" * 135)

    for task_id, vids_dict in det_data.items():
        if wanted_tasks is not None and str(task_id) not in wanted_tasks:
            continue
        recipe = id_activity.get(str(task_id), f"Task {task_id}")
        vids_shown = 0

        for vid, seg_list in vids_dict.items():
            if wanted_vids is not None and vid not in wanted_vids:
                continue
            if max_videos_per_task > 0 and vids_shown >= max_videos_per_task:
                break
            vids_shown += 1

            print(f"\n>>> Video: {vid:<8} | Recipe [{task_id}]: {recipe}")
            print(
                f"  {'Seg':<4} {'Act Range':<12} {'w/BG Dir':<10} {'Picked':<10} {'Pos%':>6} | "
                f"{'Final F':>7} {'CLIP C':>7} {'Clar Q':>7} | {'Diagnostic Flags':<34} | {'Action Step'}"
            )
            print("  " + "-" * 131)

            action_seg_idx = 0
            vis_cards = []
            for i, seg in enumerate(seg_list):
                if seg.get("action") == "BG" or not seg.get("img_path"):
                    continue
                action_seg_idx += 1
                total_segs += 1

                ts = seg["time_stamp"]
                act_s, act_e = float(ts[0]), float(ts[1])
                img_rel = seg["img_path"]  # e.g. effect_frames/1_7/7_50/000037_0.87_1.00_0.58.jpg
                parts = Path(img_rel).parts
                seg_folder = parts[-2] if len(parts) >= 2 else f"{int(math.ceil(act_s))}_{int(math.floor(act_e))}"
                fname = parts[-1]
                parsed = _parse_fname_scores(seg_folder, os.path.join(data_root, img_rel), fname)

                try:
                    m_s, m_e = [int(x) for x in seg_folder.split("_")]
                except ValueError:
                    m_s, m_e = int(math.ceil(act_s)), int(math.floor(act_e))

                f_idx = parsed["frame_idx"] or m_s
                F_val = parsed["F"] if parsed["F"] is not None else 0.0
                C_val = parsed["C"] if parsed["C"] is not None else 0.0
                Q_val = parsed["Q"] if parsed["Q"] is not None else 0.0

                flags, rel_pos = diagnose_selection_flags(f_idx, act_s, act_e, m_e, C_val, Q_val)
                rel_positions.append(rel_pos)

                if any(f.startswith("AT_VERY_START") for f in flags):
                    very_start_count += 1
                if rel_pos < 0.30 and (act_e - act_s) >= 6.0:
                    early_30_count += 1
                if f_idx > act_e:
                    in_bg_count += 1
                if Q_val < 0.35:
                    low_q_count += 1
                if C_val < 0.75:
                    low_c_count += 1
                if C_val >= 0.99:
                    c_is_1_count += 1
                if Q_val >= 0.99:
                    q_is_1_count += 1

                flag_str = ", ".join(flags) if flags else "OK (Valid Effect Timing)"
                if flags:
                    suspicious_examples.append({
                        "vid": vid,
                        "recipe": recipe,
                        "seg_idx": action_seg_idx,
                        "action": seg["action"],
                        "act_range": f"{act_s:.0f}-{act_e:.0f}s",
                        "folder": seg_folder,
                        "frame": f_idx,
                        "pos_pct": rel_pos * 100.0,
                        "F": F_val,
                        "C": C_val,
                        "Q": Q_val,
                        "flags": flag_str,
                        "gpt_rel": seg.get("Location Relationship", ""),
                    })

                print(
                    f"  #{action_seg_idx:02d}  {f'{act_s:.0f}-{act_e:.0f}s':<12} {seg_folder:<10} "
                    f"#{f_idx:04d}s    {rel_pos*100:5.0f}% | "
                    f"{F_val:7.2f} {C_val:7.2f} {Q_val:7.2f} | "
                    f"{flag_str[:34]:<34} | {seg['action'][:42]}"
                )

                if output_dir and os.path.isfile(parsed["path"]):
                    vis_cards.append((action_seg_idx, seg, parsed, f"{act_s:.0f}-{act_e:.0f}s", rel_pos, flag_str))

            if output_dir and vis_cards:
                _save_video_audit_grid(vid, recipe, vis_cards, output_dir)

    if total_segs == 0:
        print("No segments matched the filter criteria.")
        return

    rel_pcts = [x * 100.0 for x in rel_positions]
    mean_pos = statistics.mean(rel_pcts)
    med_pos = statistics.median(rel_pcts)
    print("\n" + "=" * 135)
    print(f" ROOT-CAUSE SUMMARY ACROSS {total_segs} ACTION SEGMENTS")
    print("=" * 135)
    print(f"  1. Total Action Segments Audited                : {total_segs}")
    print(f"  2. Picked TOO EARLY (<30% of action duration)   : {early_30_count:4d} ({early_30_count/total_segs*100:5.1f}%)  "
          f"[incl. {very_start_count} segments picked at the VERY FIRST second!]")
    print(f"  3. Picked in TRAILING BACKGROUND (>action end)  : {in_bg_count:4d} ({in_bg_count/total_segs*100:5.1f}%)  "
          f"[due to BG-merging in select_effect_frames.py lines 120-125]")
    print(f"  4. Picked BLURRY frame (Norm Clarity Q < 0.35)  : {low_q_count:4d} ({low_q_count/total_segs*100:5.1f}%)  "
          f"[because alpha=0.7 lets C=1.00 override low Q]")
    print(f"  5. Picked WEAK SEMANTIC frame (Norm C < 0.75)   : {low_c_count:4d} ({low_c_count/total_segs*100:5.1f}%)  "
          f"[because min-max stretches tiny CLIP cosine noise to [0, 1]]")
    print(f"  6. Segments where Selected Frame had C == 1.00  : {c_is_1_count:4d} ({c_is_1_count/total_segs*100:5.1f}%)")
    print(f"  7. Segments where Selected Frame had Q == 1.00  : {q_is_1_count:4d} ({q_is_1_count/total_segs*100:5.1f}%)")
    print(f"  8. Average Relative Timestamp within Action     : {mean_pos:.1f}% (Median: {med_pos:.1f}%)")
    print("=" * 135)


def _save_video_audit_grid(vid, recipe, vis_cards, output_dir):
    """Save a multi-row visual sheet showing all pre-extracted Effect Frames for `vid` with BBoxes & flags."""
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    card_w, card_h = 420, 250
    hdr_h = 95
    cols = min(3, len(vis_cards))
    rows = int(math.ceil(len(vis_cards) / cols))

    rendered = []
    for seg_idx, seg, parsed, act_range, rel_pos, flag_str in vis_cards:
        img = cv2.imread(parsed["path"])
        if img is None:
            continue
        img = cv2.resize(img, (card_w, card_h))

        # Draw GroundingDINO BBoxes for OBJECT A (green) and OBJECT B (cyan) if present
        for obj_key, box_key, col in [
            ("OBJECT A", "BBox of OBJECT A", (50, 220, 50)),
            ("OBJECT B", "BBox of OBJECT B", (255, 200, 50)),
        ]:
            binfo = seg.get(box_key) or {}
            bbox = binfo.get("bbox")
            prob = binfo.get("prob")
            obj_name = seg.get(obj_key, obj_key)
            if bbox and len(bbox) == 4:
                cx, cy, bw, bh = bbox
                x1 = int((cx - bw / 2) * card_w)
                y1 = int((cy - bh / 2) * card_h)
                x2 = int((cx + bw / 2) * card_w)
                y2 = int((cy + bh / 2) * card_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lbl = f"{obj_name[:18]} ({prob:.2f})" if prob else obj_name[:18]
                cv2.putText(img, lbl, (max(4, x1), max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

        is_ok = flag_str.startswith("OK")
        border_col = (60, 190, 75) if is_ok else (50, 70, 230)
        card = np.zeros((card_h + hdr_h, card_w, 3), dtype=np.uint8)
        card[:hdr_h, :] = (26, 26, 32)
        cv2.rectangle(card, (0, 0), (card_w - 1, hdr_h - 1), border_col, 2)
        card[hdr_h:, :] = img

        cv2.putText(
            card,
            f"Seg #{seg_idx:02d} [{act_range}]: {seg['action'][:38]}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            card,
            f"Picked #{parsed['frame_idx']:04d}s (Pos: {rel_pos*100:.0f}%) | F={parsed['F']:.2f} C={parsed['C']:.2f} Q={parsed['Q']:.2f}",
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 230, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            card,
            f"Status: {flag_str[:46]}",
            (8, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            border_col,
            1,
            cv2.LINE_AA,
        )
        rel_txt = (seg.get("Location Relationship") or "")[:54]
        cv2.putText(card, f"GPT4o: {rel_txt}", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
        rendered.append(card)

    if not rendered:
        return
    blank = np.zeros_like(rendered[0])
    while len(rendered) < rows * cols:
        rendered.append(blank)

    grid_rows = [np.hstack(rendered[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)
    out_path = os.path.join(output_dir, f"audit_{vid}_{recipe.replace(' ', '_')}.jpg")
    cv2.imwrite(out_path, grid)
    print(f"  [Saved Visual Audit Sheet] {out_path}")


def save_segment_comparison_image(seg_idx, vid, action, start_time, end_time,
                                  scored_rows, paper_ref, output_dir):
    """Save a visual comparison strip: Top-1 Effect Frame vs Top-2 vs Most Blurry vs Last Frame."""
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
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
    banner_h = 46
    canvas = np.zeros((strip.shape[0] + banner_h, strip.shape[1], 3), dtype=np.uint8)
    canvas[:banner_h, :] = (18, 18, 22)
    canvas[banner_h:, :] = strip
    banner_text = f"[{vid}] Seg #{seg_idx:02d} ({start_time}s - {end_time}s): {action}"
    cv2.putText(canvas, banner_text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    out_file = os.path.join(output_dir, f"{vid}_seg{seg_idx:02d}_{start_time}_{end_time}.jpg")
    cv2.imwrite(out_file, canvas)
    return out_file


def inspect_single_video_full_scoring(
    args, task_id, video_id, vid_info, recipe, idx2action, clip_bundle
):
    """Run frame-by-frame EVA02-CLIP + Laplacian scoring on `video_id`."""
    import cv2
    import numpy as np

    action_list = [idx2action[task_id][a] if a != 0 else "BG" for a in vid_info["labels"]]
    action_bounds = [[s / 10.0, e / 10.0] for s, e in vid_info["segments"]]
    raw_bounds = [list(b) for b in action_bounds]
    for i in range(len(action_list) - 1, -1, -1):
        if action_list[i] == "BG":
            if i > 0:
                action_bounds[i - 1][1] = action_bounds[i][1]
            del action_list[i]
            del action_bounds[i]
            del raw_bounds[i]

    print("\n" + "=" * 115)
    print(f" EFFECT FRAME SAMPLING INSPECTOR | Recipe [{task_id}]: {recipe} | Video: {video_id}")
    print(f" Formula: F_t = {args.alpha:.2f} * C_norm(EVA02-CLIP) + {1.0 - args.alpha:.2f} * Q_norm(Laplacian Clarity)")
    print("=" * 115)

    src_type, src_path = find_video_source(args.video_root, video_id)
    effect_frames_root = os.path.join(args.data_root, "effect_frames")

    if src_type is None:
        print(f"  [!] Raw video not found for '{video_id}' under {args.video_root} -> falling back to paper audit.")
        run_paper_audit(args.data_root, task_filter=task_id, video_filter=video_id, output_dir=args.output_dir)
        return

    print(f"  [Source] Found {src_type}: {src_path}")
    frames_dict = load_1fps_frames(src_type, src_path)
    print(f"  [Frames] Loaded {len(frames_dict)} frames sampled at 1 FPS.")

    sorted_sec = sorted(frames_dict.keys())
    vols = [laplacian_var(cv2.cvtColor(frames_dict[s], cv2.COLOR_BGR2GRAY)) for s in sorted_sec]
    p5, p95 = np.percentile(vols, [5, 95])
    print(f"  [Clarity] Global Laplacian Variance across video: P5 = {p5:.2f}, P95 = {p95:.2f}")

    recipe_desc, prompt_source = build_effect_descriptions(
        args.data_root, recipe, video_id, action_list, gpt_api_key=args.api_key
    )
    print(f"  [Prompts] Source: {prompt_source}")

    model, preprocess, tokenizer, device = clip_bundle
    num_segs = len(action_list) if args.max_segments <= 0 else min(len(action_list), args.max_segments)
    print(f"\nEvaluating {num_segs}/{len(action_list)} action segments in video '{video_id}'...\n")

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
            continue

        c_arr = np.array([r["c_raw"] for r in rows])
        q_arr = np.array([r["q_global"] for r in rows])
        c_spread = float(c_arr.max() - c_arr.min())
        c_norm = (c_arr - c_arr.min()) / (c_spread + 1e-9)
        q_norm = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min() + 1e-9)
        f_arr = args.alpha * c_norm + (1.0 - args.alpha) * q_norm

        for i, r in enumerate(rows):
            r["C"] = float(c_norm[i])
            r["Q"] = float(q_norm[i])
            r["F"] = float(f_arr[i])

        ranked = sorted(rows, key=lambda x: x["F"], reverse=True)
        for rank_i, r in enumerate(ranked, start=1):
            r["rank"] = rank_i

        best_row = ranked[0]
        last_row = rows[-1]
        paper_ref = parse_existing_effect_frame(effect_frames_root, video_id, start_time, end_time)
        flags, rel_pos = diagnose_selection_flags(
            best_row["frame_idx"], raw_b[0], raw_b[1], end_time, best_row["C"], best_row["Q"]
        )
        if c_spread < 0.025:
            flags.append(f"FLAT_CLIP(Δc={c_spread:.3f})")

        print("=" * 115)
        print(f"SEGMENT #{seg_idx:02d}: {action}")
        print(f"  Time Range : [{raw_b[0]:.1f}s -> {raw_b[1]:.1f}s] (w/ trailing BG: {bounds[1]:.1f}s) | "
              f"Raw CLIP range: [{c_arr.min():.4f}..{c_arr.max():.4f}] (Δc = {c_spread:.4f})")
        print("  CLIP Prompts:")
        for p_i, p_str in enumerate(prompts, start=1):
            print(f"    ({p_i}) \"{p_str}\"")
        print("-" * 115)
        print(f"  {'Frame (sec)':<13} | {'Raw CLIP':>9} | {'Lapl.Var':>9} | {'Clar(P5-95)':>11} | "
              f"{'Norm C':>8} | {'Norm Q':>8} | {'Final F':>9} | {'Rank':>5} | {'Note'}")
        print("-" * 115)

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
                notes.append("<-- SELECTED TOP-1")
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

        print("-" * 115)
        print(
            f"  => SELECTED EFFECT FRAME : #{best_row['frame_idx']:06d} ({best_row['frame_idx']}s, Pos={rel_pos*100:.0f}%) "
            f"| F={best_row['F']:.2f} (C={best_row['C']:.2f}, Q={best_row['Q']:.2f}) | Flags: {', '.join(flags) if flags else 'OK'}"
        )
        if paper_ref:
            p_flags, p_pos = diagnose_selection_flags(
                paper_ref["frame_idx"] or start_time, raw_b[0], raw_b[1], end_time, paper_ref["C"], paper_ref["Q"]
            )
            print(
                f"  => PAPER PRE-EXTRACTED   : #{paper_ref['frame_idx']:06d} ({paper_ref['frame_idx']}s, Pos={p_pos*100:.0f}%) "
                f"| F={paper_ref['F']:.2f} (C={paper_ref['C']:.2f}, Q={paper_ref['Q']:.2f}) | Flags: {', '.join(p_flags) if p_flags else 'OK'}"
            )

        if args.output_dir:
            saved_img = save_segment_comparison_image(
                seg_idx, video_id, action, start_time, end_time, rows, paper_ref, args.output_dir
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

    if summary_rows:
        print("=" * 115)
        print(f" SUMMARY OF SELECTED EFFECT FRAMES ({video_id} - {recipe})")
        print("=" * 115)
        print(f"{'Seg':<4} {'Range':<10} {'Selected (Ours)':<32} {'Last Frame Baseline':<32} {'Paper Pre-Extracted':<32}")
        print("-" * 115)
        for s in summary_rows:
            our_str = f"#{s['our_frame']:04d}s | {s['our_scores']}"
            last_str = f"#{s['last_frame']:04d}s | {s['last_scores']}"
            pap_str = f"#{s['paper_frame']:04d}s | {s['paper_scores']}" if s["paper_frame"] else "N/A"
            print(f"#{s['seg_idx']:02d}  {s['range']:<10} {our_str:<32} {last_str:<32} {pap_str:<32}")
        print("=" * 115)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect & Audit the paper's Effect Frame Sampling across single or ALL videos."
    )
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data", "captaincook4d"),
                        help="Path to data/captaincook4d")
    parser.add_argument("--video_root", default="/content/drive/MyDrive/AEM/datasets/CaptainCook4D",
                        help="Folder containing raw .mp4 videos or <vid>_360p 1fps frames")
    parser.add_argument("--task", type=str, default="1",
                        help="Recipe/task ID (e.g. '1', '1,10,12', or 'all')")
    parser.add_argument("--video_id", type=str, default="1_7",
                        help="Video ID (e.g. '1_7', comma-separated list, or 'all' to run on all videos in task)")
    parser.add_argument("--all_videos", action="store_true",
                        help="Run across ALL videos in the selected --task (or all recipes if --task all)")
    parser.add_argument("--audit_paper", action="store_true",
                        help="Fast audit of ALL pre-extracted Effect Frames in the dataset (diagnoses early/BG/blurry picks)")
    parser.add_argument("--max_videos_per_task", type=int, default=0,
                        help="Max videos to process per recipe when --all_videos or --audit_paper is used (0 = all)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weighting factor alpha in F = alpha * C + (1 - alpha) * Q (paper default: 0.7)")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Max number of action segments per video to print in full detail (0 = all segments)")
    parser.add_argument("--show_all_frames", action="store_true",
                        help="Print every single second/frame in each segment")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory to save comparison strips or visual audit sheets (.jpg)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Optional OpenAI API key if you want to generate prompts live with gpt-4o-mini")
    args = parser.parse_args()

    # Mode 1: Fast global audit of paper's pre-extracted Effect Frames
    if args.audit_paper:
        vid_filter = "all" if (args.all_videos or args.video_id == "1_7") else args.video_id
        run_paper_audit(
            args.data_root,
            task_filter=args.task,
            video_filter=vid_filter,
            max_videos_per_task=args.max_videos_per_task,
            output_dir=args.output_dir,
        )
        return

    # Mode 2: Full EVA02-CLIP + Laplacian frame-by-frame scoring on one or multiple videos
    non_err = read_json(os.path.join(args.data_root, "non_error_samples_processed.json"))
    err_path = os.path.join(args.data_root, "error_samples_processed.json")
    err_annot = read_json(err_path) if os.path.isfile(err_path) else {}
    id_activity = read_json(os.path.join(args.data_root, "id_activity_mapping.json"))
    activity_steps = read_json(os.path.join(args.data_root, "activity_step_collection.json"))
    step_id_map = read_json(os.path.join(args.data_root, "step_id_mapping.json"))
    idx2action = ccp4d_remapped_idx2action(id_activity, activity_steps, step_id_map)

    wanted_tasks = list(id_activity.keys()) if args.task.lower() == "all" else [t.strip() for t in args.task.split(",")]
    run_all_vids = args.all_videos or (args.video_id.lower() == "all")
    wanted_vids = None if run_all_vids else {v.strip() for v in args.video_id.split(",") if v.strip()}

    # Collect (task_id, video_id, vid_info, recipe)
    jobs = []
    for tid in wanted_tasks:
        if tid not in id_activity:
            continue
        recipe = id_activity[tid]
        merged_vids = {}
        merged_vids.update(non_err.get(tid, {}))
        merged_vids.update(err_annot.get(tid, {}))
        count = 0
        for vid, vinfo in merged_vids.items():
            if wanted_vids is not None and vid not in wanted_vids:
                continue
            jobs.append((tid, vid, vinfo, recipe))
            count += 1
            if args.max_videos_per_task > 0 and count >= args.max_videos_per_task:
                break

    if not jobs:
        raise SystemExit(f"No videos matched --task={args.task} --video_id={args.video_id}")

    print(f"Loading EVA02-CLIP-L-14-336 once for {len(jobs)} video(s)...")
    clip_bundle = load_eva_clip()

    for tid, vid, vinfo, recipe in jobs:
        inspect_single_video_full_scoring(args, tid, vid, vinfo, recipe, idx2action, clip_bundle)


if __name__ == "__main__":
    main()
