"""Component-by-Component Ablation Study for AEM (CaptainCook4D & EgoPER).

Important Architectural Distinction (Why Inference Ablation Works):
  - At TRAINING time: Effect Frames + GroundingDINO crops (`visual_info`) and GPT-4o
    Scene Graphs (`scene_graph` + GAT) act as TEACHERS via MSE/Contrastive losses
    (`effect_stt_loss`, `effect_rel_loss`).
  - What they train is the student module `Effect_Model.feature_enhancer`, which
    remains ACTIVE at INFERENCE time (`Effect_Model.inference`, lines 557-575 of
    `libs/modeling/effect_model.py`).
  - Specifically, at inference, for every predicted action segment (where label > 0):
      1. `effect_transformer` (with `CLS_token` + `pos_encoding`) reads the segment's
         frame features and outputs `effect_tokens = transformed[:, 0, :]`.
      2. `state_effect_proj` predicts the State Effect vector `e_s` (512-d).
      3. `relation_effect_proj` predicts the Spatial Relation Effect vector `e_r` (512-d).
      4. `enhanced_segment_proj` fuses `[segment_features, e_s, e_r]` -> enhanced 512-d.
      5. `Action_Model` encodes action text prompts with a 4-token `prompt_prefix` and
         computes cosine similarity against the enhanced segment feature to detect mistakes.

This script systematically ablates each of these active inference-time components
(plus NMS post-processing and Oracle GT segment boundaries) across checkpoints and
prints a unified comparison table with deltas (Δ) relative to the Full AEM model.
"""

import argparse
import glob
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
UTILS_DIR = os.path.join(REPO_ROOT, "libs", "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.datasets.clip_encode_ccp4d import action_tokenize_ccp4d
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed

from test_segmentation import (
    CombinedCaptainCook4DDataset,
    _resolve_root_dir,
    compute_ed_metrics,
    compute_extended_seg_metrics,
    load_clip_model,
    load_split_video_ids,
    load_weights,
)


@contextmanager
def ablate_model_context(model, mode: str):
    """Temporarily patch `model` according to the requested ablation `mode`.

    Note: `mode` may optionally be prefixed with `'oracle_'` (e.g., `'oracle_no_rel_effect'`,
    `'oracle_bypass_effect'`), which strips the prefix here so the exact same component
    ablation is applied on top of Ground-Truth (Oracle) segment boundaries & labels.

    Supported component modes (both Predicted and `oracle_*`):
      - 'full'            : Original unmodified AEM pipeline (Paper Baseline).
      - 'no_rel_effect'   : Zero out `relation_effect_token` (keep only State Effect `e_s`).
      - 'no_state_effect' : Zero out `state_effect_token` (keep only Spatial Relation `e_r`).
      - 'zero_both_tokens': Run `effect_transformer` on segment frames, but zero out both
                            `e_s` and `e_r` before `enhanced_segment_proj`.
      - 'bypass_effect'   : Bypass `effect_model.inference` completely (pass raw ActionFormer
                            FPN segment features directly to `action_model`).
      - 'no_pos_enc'      : Disable sinusoidal `pos_encoding` inside `Effect_Model.feature_enhancer`.
      - 'no_prompt_prefix': Disable the 4 learnable prefix tokens `action_model.prompt_prefix`
                            (use plain frozen EVA02-CLIP text embeddings).
      - 'bypass_and_no_prefix': Bypass `effect_model.inference` AND disable `prompt_prefix`
                            (pure ActionFormer FPN features + frozen EVA02-CLIP text).
      - 'tuned_nms'       : Increase `test_min_score` (0.001 -> 0.05) and reduce `test_max_seg_num`
                            (200 -> 40) to suppress severe over-segmentation (Predicted Segs only).
    """
    base_mode = mode[len("oracle_"):] if mode.startswith("oracle_") else mode
    if base_mode == "gt_segs":
        base_mode = "full"

    fe = model.effect_model.feature_enhancer
    am = model.action_model

    orig_enhance = fe.enhance_feature
    orig_ef_infer = model.effect_model.inference
    orig_pos_enc = fe.pos_encoding
    orig_prefix = am.prompt_prefix
    orig_min_score = model.test_min_score
    orig_max_seg = model.test_max_seg_num

    try:
        if base_mode in ("no_rel_effect", "no_state_effect", "zero_both_tokens"):
            def patched_enhance(feature_segments, effect_model_mask, device):
                effect_segments = [
                    feature_segments[i] for i in range(len(feature_segments)) if effect_model_mask[i]
                ]
                if not effect_segments:
                    return feature_segments, None, None

                padded_segments = pad_sequence(effect_segments, batch_first=True)
                b, _, _ = padded_segments.shape
                cls_tokens = fe.CLS_token.expand(b, -1, -1).to(device)
                padded_segments = torch.cat([cls_tokens, padded_segments], dim=1)

                if fe.pos_encoding is not None:
                    padded_segments = fe.pos_encoding(padded_segments)

                lengths = torch.tensor([feat.shape[0] for feat in effect_segments], device=device)
                max_length = padded_segments.size(1)
                memory_mask = (
                    torch.arange(max_length, device=device).expand(len(lengths), max_length)
                    >= (lengths + 1).unsqueeze(1)
                )

                transformed = fe.effect_transformer(padded_segments, src_key_padding_mask=memory_mask)
                effect_tokens = transformed[:, 0, :]

                state_effect_token = fe.state_effect_proj(effect_tokens)
                relation_effect_token = fe.relation_effect_proj(effect_tokens)

                if base_mode in ("no_state_effect", "zero_both_tokens"):
                    state_effect_token = torch.zeros_like(state_effect_token)
                if base_mode in ("no_rel_effect", "zero_both_tokens"):
                    relation_effect_token = torch.zeros_like(relation_effect_token)

                segment_features = transformed[:, 1:, :]
                seq_len = segment_features.size(1)
                state_expanded = state_effect_token.unsqueeze(1).expand(-1, seq_len, -1)
                relation_expanded = relation_effect_token.unsqueeze(1).expand(-1, seq_len, -1)
                enhanced = torch.cat([segment_features, state_expanded, relation_expanded], dim=-1)
                enhanced = fe.enhanced_segment_proj(enhanced)

                enhanced_effect_segments = [enhanced[i, :lengths[i], :] for i in range(len(lengths))]
                enhanced_iter = iter(enhanced_effect_segments)
                enhanced_segments = [
                    next(enhanced_iter) if effect_model_mask[i] else feature_segments[i]
                    for i in range(len(feature_segments))
                ]
                return enhanced_segments, state_effect_token, relation_effect_token

            fe.enhance_feature = patched_enhance

        elif base_mode == "bypass_effect":
            def bypass_infer(feature_segments, action_labels, device):
                return feature_segments
            model.effect_model.inference = bypass_infer

        elif base_mode == "no_pos_enc":
            fe.pos_encoding = None

        elif base_mode == "no_prompt_prefix":
            am.prompt_prefix = None

        elif base_mode == "bypass_and_no_prefix":
            def bypass_infer(feature_segments, action_labels, device):
                return feature_segments
            model.effect_model.inference = bypass_infer
            am.prompt_prefix = None

        elif base_mode == "tuned_nms":
            model.test_min_score = 0.05
            model.test_max_seg_num = 40

        yield

    finally:
        fe.enhance_feature = orig_enhance
        model.effect_model.inference = orig_ef_infer
        fe.pos_encoding = orig_pos_enc
        am.prompt_prefix = orig_prefix
        model.test_min_score = orig_min_score
        model.test_max_seg_num = orig_max_seg


@torch.no_grad()
def run_ablation_mode(
    model,
    test_loader,
    clip_model,
    action_tokens,
    mode: str,
    cached_videos=None,
):
    """Run inference on `test_loader` (or pre-extracted `cached_videos`) under `mode`."""
    all_gt_frames = []
    all_pred_frames = []
    ed_results = {}
    use_oracle_segs = mode.startswith("oracle_")

    with ablate_model_context(model, mode):
        iterable = cached_videos if cached_videos is not None else test_loader
        for item in iterable:
            if cached_videos is not None:
                video_list = item["video_list"]
                fpn_feats = item["fpn_feats"]
                fpn_masks = item["fpn_masks"]
                points = item["points"]
                cls_prob_logits = item["cls_prob_logits"]
                out_offsets = item["out_offsets"]
                base_seg_results = item["base_seg_results"]
            else:
                video_list = item
                fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = model.extract_feature(video_list)
                base_seg_results = None

            if mode == "tuned_nms" or base_seg_results is None:
                results = model.segment_video(video_list, points, fpn_masks, cls_prob_logits, out_offsets)
            else:
                results = {
                    vid: {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in res.items()}
                    for vid, res in base_seg_results.items()
                }

            video_id, vid_result = list(results.items())[0]

            if use_oracle_segs:
                # Use Ground-Truth segment boundaries & labels instead of ActionFormer predictions
                vinfo = video_list[0]
                gt_segs = vinfo["segments"].cpu()
                gt_labels = vinfo["labels"].cpu()

                segment_feature = model.generate_segment_features(fpn_feats, fpn_masks, [gt_segs.to(model.device)])
                enhanced_segment_feature = model.effect_model.inference(
                    feature_segments=segment_feature,
                    action_labels=[gt_labels],
                    device=model.device,
                )
                segment_vis_feature = torch.stack([seg.mean(dim=0) for seg in enhanced_segment_feature], dim=0)
                action_similarity = model.action_model.inference(
                    clip_model=clip_model,
                    segment_vis_feature=segment_vis_feature,
                    action_tokens=action_tokens,
                    action_labels=[gt_labels],
                    device=model.device,
                )
                repeat_num = (gt_segs[:, 1] - gt_segs[:, 0] + 1).long().clamp(min=1)
                vid_result["segments"] = gt_segs
                vid_result["segment_action_pred"] = gt_labels
                vid_result["frame_action_pred"] = vid_result["frame_action_label"].clone()
                vid_result["framewise_action_sim"] = torch.repeat_interleave(
                    action_similarity.cpu(), repeats=repeat_num
                )
                vid_result["segment_action_sim"] = action_similarity.cpu()
            else:
                action_segments = [vid_result["segments"]]
                segment_feature = model.generate_segment_features(fpn_feats, fpn_masks, action_segments)
                enhanced_segment_feature = model.effect_model.inference(
                    feature_segments=segment_feature,
                    action_labels=[vid_result["segment_action_pred"]],
                    device=model.device,
                )
                segment_vis_feature = torch.stack([seg.mean(dim=0) for seg in enhanced_segment_feature], dim=0)
                action_similarity = model.action_model.inference(
                    clip_model=clip_model,
                    segment_vis_feature=segment_vis_feature,
                    action_tokens=action_tokens,
                    action_labels=[vid_result["segment_action_pred"]],
                    device=model.device,
                )
                for vid, result in results.items():
                    repeat_num = (result["segments"][:, 1] - result["segments"][:, 0] + 1).long().clamp(min=1)
                    result["framewise_action_sim"] = torch.repeat_interleave(
                        action_similarity.cpu(), repeats=repeat_num
                    )
                    result["segment_action_sim"] = action_similarity.cpu()

            for vid, res in results.items():
                all_gt_frames.append(res["frame_action_label"].clone().cpu().numpy())
                all_pred_frames.append(res["frame_action_pred"].clone().cpu().numpy())
                ed_results[vid] = res

    metrics_with_bg = compute_extended_seg_metrics(all_gt_frames, all_pred_frames, ignore_bg=False)
    metrics_no_bg = compute_extended_seg_metrics(all_gt_frames, all_pred_frames, ignore_bg=True)
    ed_metrics = compute_ed_metrics(ed_results)
    return {
        "with_bg": metrics_with_bg,
        "without_bg": metrics_no_bg,
        "ed_metrics": ed_metrics,
    }


PRED_ABLATION_MODES = [
    ("full",                 "1. Full AEM Model (Paper Baseline)"),
    ("no_rel_effect",        "2. w/o Spatial Relation Effect (e_r = 0, State Only)"),
    ("no_state_effect",      "3. w/o Object State Effect (e_s = 0, Relation Only)"),
    ("zero_both_tokens",     "4. w/o Both Effect Tokens (e_s = 0 & e_r = 0)"),
    ("bypass_effect",        "5. Bypass Effect_Model Completely (Raw FPN Seg Feat)"),
    ("no_pos_enc",           "6. w/o Effect Transformer Positional Encoding"),
    ("no_prompt_prefix",     "7. w/o Learnable Prompt Prefix (Frozen CLIP Text)"),
    ("bypass_and_no_prefix", "8. Bypass Effect_Model + w/o Prompt Prefix"),
    ("tuned_nms",            "9. Tuned NMS Post-Processing (min_score=0.05, max=40)"),
]

ORACLE_ABLATION_MODES = [
    ("oracle_full",                 "10. [Oracle GT Segs] Full AEM (Effect + Action_Model)"),
    ("oracle_no_rel_effect",        "11. [Oracle GT Segs] w/o Spatial Relation (e_r = 0)"),
    ("oracle_no_state_effect",      "12. [Oracle GT Segs] w/o Object State (e_s = 0)"),
    ("oracle_zero_both_tokens",     "13. [Oracle GT Segs] w/o Both Tokens (e_s=0 & e_r=0)"),
    ("oracle_bypass_effect",        "14. [Oracle GT Segs] Bypass Effect_Model Completely"),
    ("oracle_no_pos_enc",           "15. [Oracle GT Segs] w/o Effect Positional Encoding"),
    ("oracle_no_prompt_prefix",     "16. [Oracle GT Segs] w/o Learnable Prompt Prefix"),
    ("oracle_bypass_and_no_prefix", "17. [Oracle GT Segs] Bypass Effect + w/o Prompt Prefix"),
]

ABLATION_MODES = PRED_ABLATION_MODES + ORACLE_ABLATION_MODES


def main():
    parser = argparse.ArgumentParser(
        description="Run component-by-component ablation study on AEM checkpoints."
    )
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ccp4d"))
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs (e.g. '1,10,12') or 'all' for all available recipes")
    parser.add_argument("--split_file", type=str,
                        default=os.path.join(TESTS_DIR, "person_data_split_combined.json"),
                        help="Path to split JSON, or 'none' for paper's original error-only test set")
    parser.add_argument("--split_subset", type=str, default="test")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--only_oracle", action="store_true",
                        help="Only run the 8 Oracle Ground-Truth Segmentation ablation modes")
    args = parser.parse_args()

    active_modes = ORACLE_ABLATION_MODES if args.only_oracle else ABLATION_MODES

    _ = fix_random_seed(0, include_cuda=True)
    split_file = None if (args.split_file and args.split_file.lower() == "none") else args.split_file

    cfg_base = load_config(args.config)
    root_dir = _resolve_root_dir(cfg_base)
    device = cfg_base["devices"][0]

    with open(os.path.join(root_dir, "id_activity_mapping.json"), "r") as fp:
        id_activity = json.load(fp)
    with open(os.path.join(root_dir, "activity_step_collection.json"), "r") as fp:
        activity_step_collection = json.load(fp)
    with open(os.path.join(root_dir, "step_id_mapping.json"), "r") as fp:
        step_id_mapping = json.load(fp)

    wanted_tasks = None
    if args.tasks.lower() != "all":
        wanted_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    matched_ckpts = []
    for task_id, activity_name in id_activity.items():
        if wanted_tasks is not None and str(task_id) not in wanted_tasks:
            continue
        pattern = os.path.join(args.ckpt_dir, f"recipe_{task_id}_*.pth.tar")
        matches = sorted(glob.glob(pattern))
        if matches:
            matched_ckpts.append((str(task_id), activity_name, matches[0]))

    if not matched_ckpts:
        raise FileNotFoundError(f"No matching checkpoints found in {args.ckpt_dir} (tasks={args.tasks})")

    print("=" * 132)
    print(
        f" AEM COMPONENT ABLATION STUDY | Tasks: {len(matched_ckpts)} | "
        f"Split: {os.path.basename(split_file) + '[' + args.split_subset + ']' if split_file else 'original (error_samples_processed.json)'}"
    )
    print("=" * 132)
    print("Loading EVA02-CLIP-L-14-336 for Mistake Detection & Prompt Prefix evaluation...")
    clip_model = load_clip_model(device)

    mode_task_results = {m_key: [] for m_key, _ in ABLATION_MODES}

    for task_id, recipe_name, ckpt_path in matched_ckpts:
        cfg = load_config(args.config)
        cfg["dataset"]["task"] = task_id
        root_dir = _resolve_root_dir(cfg)

        if split_file and cfg["dataset_name"] == "CaptainCook4D":
            allowed_vids = load_split_video_ids(split_file, subset=args.split_subset, task=task_id)
            test_dataset = CombinedCaptainCook4DDataset(
                root_dir=root_dir,
                task=task_id,
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
                cfg["dataset_name"], False, None, cfg["test_split"], False, **cfg["dataset"]
            )
            test_loader = make_data_loader(
                test_dataset, False, None, 1, cfg["loader"]["num_workers"]
            )

        if len(test_dataset) == 0:
            print(f"  [Skip] Task {task_id} ({recipe_name}): 0 test videos matched.")
            continue

        action_tokens, num_classes = action_tokenize_ccp4d(
            clip_model, task_id, activity_step_collection, step_id_mapping, device=device
        )
        cfg["model"]["num_classes"] = num_classes
        cfg["model"]["backbone_type"] = "convTransformer"

        model = make_meta_arch(cfg["model_name"], **cfg["model"]).to(device)
        model = load_weights(model, ckpt_path, use_ema=args.use_ema)
        model.eval()

        # Pre-extract ActionFormer FPN features & default segmentation once per video
        # so evaluating 17 ablation modes is ~10x faster and bit-exact.
        cached_videos = []
        with torch.no_grad():
            for video_list in test_loader:
                fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = model.extract_feature(video_list)
                base_seg_results = model.segment_video(
                    video_list, points, fpn_masks, cls_prob_logits, out_offsets
                )
                cached_videos.append({
                    "video_list": video_list,
                    "fpn_feats": fpn_feats,
                    "fpn_masks": fpn_masks,
                    "points": points,
                    "cls_prob_logits": cls_prob_logits,
                    "out_offsets": out_offsets,
                    "base_seg_results": base_seg_results,
                })

        print(f"  -> Task {task_id:>2s} ({recipe_name:<31s}) | {len(test_dataset)} videos | running {len(active_modes)} ablation modes...")
        for m_key, _ in active_modes:
            res = run_ablation_mode(
                model,
                test_loader,
                clip_model,
                action_tokens,
                mode=m_key,
                cached_videos=cached_videos,
            )
            mode_task_results[m_key].append(res)

    # Compute macro averages across evaluated tasks for each mode
    summary = {}
    for m_key, m_label in active_modes:
        res_list = mode_task_results[m_key]
        if not res_list:
            continue
        w_list = [r["with_bg"] for r in res_list]
        nb_list = [r["without_bg"] for r in res_list]
        ed_list = [r["ed_metrics"] for r in res_list if r["ed_metrics"] is not None]

        tot_pred_w = sum(x["pred_segs"] for x in w_list)
        tot_gt_w = sum(x["gt_segs"] for x in w_list)

        summary[m_key] = {
            "label": m_label,
            "acc_w": float(np.mean([x["Accuracy"] for x in w_list])),
            "miou_w": float(np.mean([x["mIoU"] for x in w_list])),
            "edit_w": float(np.mean([x["Edit"] for x in w_list])),
            "f1_50_w": float(np.mean([x["F1@50"] for x in w_list])),
            "frag_w": tot_pred_w / max(1, tot_gt_w),
            "acc_nb": float(np.mean([x["Accuracy"] for x in nb_list])),
            "f1_50_nb": float(np.mean([x["F1@50"] for x in nb_list])),
            "prec": float(np.mean([x["precision"] for x in ed_list])) if ed_list else 0.0,
            "auc": float(np.mean([x["auc"] for x in ed_list])) if ed_list else 0.0,
            "eda": float(np.mean([x["eda"] for x in ed_list])) if ed_list else 0.0,
        }

    base_oracle = summary.get("oracle_full")
    base_pred = summary.get("full", base_oracle)
    num_eval_recipes = len(mode_task_results[active_modes[0][0]])

    if not args.only_oracle and "full" in summary:
        print("\n" + "=" * 138)
        print(f" TABLE A: ABLATION ON PREDICTED SEGMENTS (ActionFormer Segs) ACROSS {num_eval_recipes} RECIPES")
        print("=" * 138)
        header_a = (
            f"{'Ablation Variant (Predicted Segmentation)':<54} | "
            f"{'Acc(w/BG)':>9} {'Edit':>6} {'F1@50':>6} {'Frag':>6} | "
            f"{'Prec':>6} {'(ΔPrec)':>8} {'AUC':>6} {'(ΔAUC)':>8} {'EDA':>6} {'(ΔEDA)':>8}"
        )
        print(header_a)
        print("-" * 138)

        for m_key, _ in PRED_ABLATION_MODES:
            if m_key not in summary:
                continue
            s = summary[m_key]
            d_prec = s["prec"] - base_pred["prec"]
            d_auc = s["auc"] - base_pred["auc"]
            d_eda = s["eda"] - base_pred["eda"]
            d_prec_str = f"{d_prec:+6.2f}%" if m_key != "full" else "  base "
            d_auc_str = f"{d_auc:+6.2f}%" if m_key != "full" else "  base "
            d_eda_str = f"{d_eda:+6.2f}%" if m_key != "full" else "  base "

            if m_key == "tuned_nms":
                print("-" * 138)
            print(
                f"{s['label']:<54} | "
                f"{s['acc_w']:>8.2f}% {s['edit_w']:>6.2f} {s['f1_50_w']:>6.2f} {s['frag_w']:>5.2f}x | "
                f"{s['prec']:>6.2f} {d_prec_str:>8} {s['auc']:>6.2f} {d_auc_str:>8} {s['eda']:>6.2f} {d_eda_str:>8}"
            )

    print("\n" + "=" * 138)
    print(f" TABLE B: ABLATION ON ORACLE GROUND-TRUTH SEGMENTS (100% Perfect Segs) ACROSS {num_eval_recipes} RECIPES")
    print("=" * 138)
    header_b = (
        f"{'Ablation Variant (Oracle GT Segmentation)':<54} | "
        f"{'Prec':>6} {'(Δ vs Orcl)':>11} {'AUC':>6} {'(Δ vs Orcl)':>11} {'EDA':>6} {'(Δ vs Orcl)':>11} | "
        f"{'Gain vs Pred':>14}"
    )
    print(header_b)
    print("-" * 138)

    paired_keys = [
        ("oracle_full",                 "full"),
        ("oracle_no_rel_effect",        "no_rel_effect"),
        ("oracle_no_state_effect",      "no_state_effect"),
        ("oracle_zero_both_tokens",     "zero_both_tokens"),
        ("oracle_bypass_effect",        "bypass_effect"),
        ("oracle_no_pos_enc",           "no_pos_enc"),
        ("oracle_no_prompt_prefix",     "no_prompt_prefix"),
        ("oracle_bypass_and_no_prefix", "bypass_and_no_prefix"),
    ]
    for o_key, p_key in paired_keys:
        if o_key not in summary:
            continue
        s = summary[o_key]
        d_prec = s["prec"] - base_oracle["prec"]
        d_auc = s["auc"] - base_oracle["auc"]
        d_eda = s["eda"] - base_oracle["eda"]
        d_prec_str = f"{d_prec:+6.2f}%" if o_key != "oracle_full" else "    base   "
        d_auc_str = f"{d_auc:+6.2f}%" if o_key != "oracle_full" else "    base   "
        d_eda_str = f"{d_eda:+6.2f}%" if o_key != "oracle_full" else "    base   "
        if p_key in summary:
            gain_str = f"AUC {s['auc'] - summary[p_key]['auc']:+6.2f}%"
        else:
            gain_str = f"{'N/A (Oracle)':>12}"

        print(
            f"{s['label']:<54} | "
            f"{s['prec']:>6.2f} {d_prec_str:>11} {s['auc']:>6.2f} {d_auc_str:>11} {s['eda']:>6.2f} {d_eda_str:>11} | "
            f"{gain_str:>14}"
        )

    print("-" * 138)
    print(
        f"{'Paper Reported Reference (ICLR 2026 Table 1 & 2)':<54} | "
        f"{'68.10':>6} {'':>11} {'62.50':>6} {'':>11} {'71.90':>6} {'':>11} |"
    )
    print("=" * 138)


if __name__ == "__main__":
    main()
