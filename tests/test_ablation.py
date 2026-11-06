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
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
UTILS_DIR = os.path.join(REPO_ROOT, "libs", "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, to_frame_wise, to_segments
from libs.modeling import make_meta_arch
from libs.modeling.action_model import PrefixPromptTextEncoder

from tests.test_segmentation import (
    CCP4D_RECIPES,
    CombinedCaptainCook4DDataset,
    _resolve_root_dir,
    build_action_tokens,
    compute_ed_metrics,
    compute_extended_metrics,
    find_ccp4d_checkpoints,
    load_checkpoint_weights,
    load_clip_model,
)


@contextmanager
def ablate_model_context(model, mode: str):
    """Temporarily patch `model` according to the requested ablation `mode`.

    Supported modes:
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
      - 'tuned_nms'       : Increase `test_min_score` (0.001 -> 0.05) and reduce `test_max_seg_num`
                            (200 -> 40) to suppress severe over-segmentation.
      - 'oracle_gt_segs'  : Use ground-truth segment boundaries & labels instead of predicted
                            segments to measure the true upper bound of Effect + Action_Model.
    """
    fe = model.effect_model.feature_enhancer
    am = model.action_model

    orig_enhance = fe.enhance_feature
    orig_ef_infer = model.effect_model.inference
    orig_pos_enc = fe.pos_encoding
    orig_prefix = am.prompt_prefix
    orig_min_score = model.test_min_score
    orig_max_seg = model.test_max_seg_num

    try:
        if mode in ("no_rel_effect", "no_state_effect", "zero_both_tokens"):
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

                if mode in ("no_state_effect", "zero_both_tokens"):
                    state_effect_token = torch.zeros_like(state_effect_token)
                if mode in ("no_rel_effect", "zero_both_tokens"):
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

        elif mode == "bypass_effect":
            def bypass_infer(feature_segments, action_labels, device):
                return feature_segments
            model.effect_model.inference = bypass_infer

        elif mode == "no_pos_enc":
            fe.pos_encoding = None

        elif mode == "no_prompt_prefix":
            am.prompt_prefix = None

        elif mode == "tuned_nms":
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
def run_ablation_mode(model, test_loader, clip_model, action_tokens, mode: str):
    """Run inference on `test_loader` under the specified ablation `mode`."""
    all_preds, all_gts = [], []
    all_pred_segs, all_gt_segs = [], []
    ed_results = {}

    with ablate_model_context(model, mode):
        for video_list in test_loader:
            if mode == "oracle_gt_segs":
                # Use Ground-Truth segment boundaries & labels instead of ActionFormer predictions
                fpn_feats, fpn_masks, points, cls_prob_logits, out_offsets = model.extract_feature(video_list)
                results = model.segment_video(video_list, points, fpn_masks, cls_prob_logits, out_offsets)
                video_id, vid_result = list(results.items())[0]
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
                results = model.inference(video_list, clip_model, action_tokens)

            for vid, res in results.items():
                pred_f = res["frame_action_pred"].cpu().numpy()
                gt_f = res["frame_action_label"].cpu().numpy()
                min_len = min(len(pred_f), len(gt_f))
                pred_f = pred_f[:min_len]
                gt_f = gt_f[:min_len]

                all_preds.append(pred_f)
                all_gts.append(gt_f)
                all_pred_segs.append(to_segments(pred_f))
                all_gt_segs.append(to_segments(gt_f))
                ed_results[vid] = res

    metrics_with_bg = compute_extended_metrics(
        all_preds, all_gts, all_pred_segs, all_gt_segs, ignore_bg=False
    )
    metrics_no_bg = compute_extended_metrics(
        all_preds, all_gts, all_pred_segs, all_gt_segs, ignore_bg=True
    )
    ed_metrics = compute_ed_metrics(ed_results)
    return {
        "with_bg": metrics_with_bg,
        "without_bg": metrics_no_bg,
        "ed_metrics": ed_metrics,
        "ed_raw": ed_results,
    }


ABLATION_MODES = [
    ("full",             "1. Full AEM Model (Paper Baseline)"),
    ("no_rel_effect",    "2. w/o Spatial Relation Effect (e_r = 0, State Only)"),
    ("no_state_effect",  "3. w/o Object State Effect (e_s = 0, Relation Only)"),
    ("zero_both_tokens", "4. w/o Both Effect Tokens (e_s = 0 & e_r = 0)"),
    ("bypass_effect",    "5. Bypass Effect_Model Completely (Raw FPN Seg Feat)"),
    ("no_pos_enc",       "6. w/o Effect Transformer Positional Encoding"),
    ("no_prompt_prefix", "7. w/o Learnable Prompt Prefix (Frozen CLIP Text)"),
    ("tuned_nms",        "8. Tuned NMS Post-Processing (min_score=0.05, max=40)"),
    ("oracle_gt_segs",   "9. Oracle GT Segments (Upper Bound of Effect+Detector)"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run component-by-component ablation study on AEM checkpoints."
    )
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ccp4d"))
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs (e.g. '1,10,12') or 'all' for all available recipes")
    parser.add_argument("--split_file", type=str,
                        default=os.path.join(REPO_ROOT, "tests", "person_data_split_combined.json"),
                        help="Path to split JSON, or 'none' for paper's original error-only test set")
    parser.add_argument("--split_subset", type=str, default="test", choices=["val", "test", "train"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    split_file = None if (args.split_file and args.split_file.lower() == "none") else args.split_file
    device = torch.device(args.device)

    ckpts = find_ccp4d_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {args.ckpt_dir}")

    if args.tasks.lower() != "all":
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        ckpts = {k: v for k, v in ckpts.items() if k in wanted}
        if not ckpts:
            raise ValueError(f"None of the requested --tasks ({args.tasks}) matched checkpoints in {args.ckpt_dir}")

    print("=" * 120)
    print(f" AEM COMPONENT ABLATION STUDY | Tasks: {len(ckpts)} | Split: {os.path.basename(split_file) if split_file else 'original'}")
    print("=" * 120)
    print("Loading EVA02-CLIP-L-14-336 for Mistake Detection & Prompt Prefix evaluation...")
    clip_model = load_clip_model(device)

    # Aggregate containers per ablation mode across all evaluated tasks
    mode_TaskResults = {m_key: [] for m_key, _ in ABLATION_MODES}

    for task_id, ckpt_path in ckpts.items():
        recipe_name = CCP4D_RECIPES.get(task_id, f"Task {task_id}")
        cfg = load_config(args.config)
        cfg["dataset"]["task"] = str(task_id)
        cfg = _resolve_root_dir(cfg)

        if split_file and os.path.isfile(split_file):
            test_dataset = CombinedCaptainCook4DDataset(
                split_Status="test",
                root_dir=cfg["dataset"]["root_dir"],
                annotation_dir=cfg["dataset"]["annotation_dir"],
                feat_Folders=cfg["dataset"]["feat_folder"],
                task=str(task_id),
                fps=cfg["dataset"]["fps"],
                split_file=split_file,
                split_subset=args.split_subset,
            )
            if len(test_dataset) == 0:
                continue
            cfg["dataset"]["num_classes"] = test_dataset.get_Wait_num_classes()
            cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
            test_loader = make_data_loader(test_dataset, False, None, 1, cfg["loader"]["num_workers"])
        else:
            test_dataset = make_dataset(cfg["dataset_name"], False, cfg["dataset"], "test")
            if len(test_dataset) == 0:
                continue
            cfg["dataset"]["num_classes"] = test_dataset.get_Wait_num_classes()
            cfg["model"]["num_classes"] = cfg["dataset"]["num_classes"]
            test_loader = make_data_loader(test_dataset, False, None, 1, cfg["loader"]["num_workers"])

        model = make_meta_arch(cfg["model_name"], **cfg["model"]).to(device)
        load_checkpoint_weights(model, ckpt_path, device)
        model.eval()

        action_tokens = build_action_tokens(cfg, clip_model, device)

        print(f"  -> Evaluating Task {task_id:>2s} ({recipe_name:<30s}) | {len(test_dataset)} videos across {len(ABLATION_MODES)} ablation modes...")
        for m_key, _ in ABLATION_MODES:
            res = run_ablation_mode(model, test_loader, clip_model, action_tokens, mode=m_key)
            mode_TaskResults[m_key].append(res)

    # Compute macro averages across evaluated tasks for each mode
    summary = {}
    for m_key, m_label in ABLATION_MODES:
        res_list = mode_TaskResults[m_key]
        if not res_list:
            continue
        w_list = [r["with_bg"] for r in res_list]
        nb_list = [r["without_bg"] for r in res_list]
        ed_list = [r["ed_metrics"] for r in res_list if r["ed_metrics"] is not None]

        tot_pred_w = sum(x["num_pred_segs"] for x in w_list)
        tot_gt_w = sum(x["num_gt_segs"] for x in w_list)

        summary[m_key] = {
            "label": m_label,
            "acc_w": float(np.mean([x["acc"] for x in w_list])),
            "miou_w": float(np.mean([x["miou"] for x in w_list])),
            "edit_w": float(np.mean([x["edit"] for x in w_list])),
            "f1_50_w": float(np.mean([x["f1@50"] for x in w_list])),
            "frag_w": tot_pred_w / max(1, tot_gt_w),
            "acc_nb": float(np.mean([x["acc"] for x in nb_list])),
            "f1_50_nb": float(np.mean([x["f1@50"] for x in nb_list])),
            "prec": float(np.mean([x["precision"] for x in ed_list])) if ed_list else 0.0,
            "auc": float(np.mean([x["auc"] for x in ed_list])) if ed_list else 0.0,
            "eda": float(np.mean([x["eda"] for x in ed_list])) if ed_list else 0.0,
        }

    base = summary["full"]

    print("\n" + "=" * 132)
    print(f" ABLATION STUDY SUMMARY ACROSS {len(mode_TaskResults['full'])} RECIPES")
    print("=" * 132)
    header = (
        f"{'Ablation Variant / Component Tested':<54} | "
        f"{'Acc(w/BG)':>9} {'Edit':>6} {'F1@50':>6} {'Frag':>6} | "
        f"{'Prec':>6} {'AUC':>6} {'(ΔAUC)':>8} {'EDA':>6} {'(ΔEDA)':>8}"
    )
    print(header)
    print("-" * 132)

    for m_key, _ in ABLATION_MODES:
        if m_key not in summary:
            continue
        s = summary[m_key]
        d_auc = s["auc"] - base["auc"]
        d_eda = s["eda"] - base["eda"]
        d_auc_str = f"{d_auc:+6.2f}%" if m_key != "full" else "  base "
        d_eda_str = f"{d_eda:+6.2f}%" if m_key != "full" else "  base "

        if m_key == "tuned_nms":
            print("-" * 132)
        print(
            f"{s['label']:<54} | "
            f"{s['acc_w']:>8.2f}% {s['edit_w']:>6.2f} {s['f1_50_w']:>6.2f} {s['frag_w']:>5.2f}x | "
            f"{s['prec']:>6.2f} {s['auc']:>6.2f} {d_auc_str:>8} {s['eda']:>6.2f} {d_eda_str:>8}"
        )

    print("-" * 132)
    print(
        f"{'Paper Reported Reference (ICLR 2026 Table 1 & 2)':<54} | "
        f"{'69.70%':>9} {'--':>6} {'--':>6} {'--':>6} | "
        f"{'68.10':>6} {'62.50':>6} {'':>8} {'71.90':>6} {'':>8}"
    )
    print("=" * 132)


if __name__ == "__main__":
    main()
