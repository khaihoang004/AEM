"""Train & Evaluate the Streamlined "No-AEM" Baseline on CaptainCook4D.

What this script trains (AEM Removed):
  - Block 1: ActionFormer Backbone + FPN + Dynamic Cross-Level Fusion Heads
             (`DynClsHead` + `DynRegHead`) trained with Focal Cls Loss + DIoU Reg Loss.
  - Block 2: Prompt-based Mistake Detector (`Action_Model`) aligning segment features
             with EVA02-CLIP-L-14-336 text prompts via Softmax Cross-Entropy (`det_ce_loss`)
             and Sigmoid Focal Loss (`det_focal_loss`).

What is completely skipped (0 overhead, no VLM/Scene-Graph/Effect-Frame needed):
  - No `effect_frames/`, no `openset_detection_ccp4d.json`, no `scene_graph.npy` loaded
    (`CCP4D_SKIP_VLM=1` is set automatically so dataset initialization takes <0.1s).
  - `Effect_Model` parameters are frozen and bypassed during both `forward()` and
    `inference()` (unless `--keep_effect_transformer` is passed to retain just the
    self-attention temporal smoothing layer without VLM supervision).
  - Supports training a single recipe (`--task 1`), a subset (`--tasks 1,10,12`), or all
    24 CaptainCook4D recipes (`--tasks all`), and automatically evaluates each trained
    model against the released paper checkpoint on both Action Segmentation and Mistake
    Detection (`Precision`, `AUC`, `EDA`).
"""

import argparse
import glob
import json
import os
import sys
import time

# Skip loading VLM detections, Effect Frames, and Scene Graphs in CaptainCook4DDataset
os.environ["CCP4D_SKIP_VLM"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn

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
from libs.utils import (
    ModelEma,
    fix_random_seed,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
)

from test_segmentation import (
    CombinedCaptainCook4DDataset,
    _resolve_root_dir,
    compute_ed_metrics,
    compute_extended_seg_metrics,
    load_clip_model,
    load_split_video_ids,
    load_weights,
    run_full_inference,
)


def configure_no_aem_model(model: nn.Module, keep_effect_transformer: bool = False):
    """Patch `PtTransformer` so `Effect_Model` (AEM) is bypassed during both training and inference.

    - If `keep_effect_transformer=False` (default Pure No-AEM):
      `Effect_Model` is completely bypassed (`enhanced_segments = segments`) and all its
      parameters have `requires_grad = False` so the optimizer only updates `backbone`,
      `neck`, `cls_head`, `reg_head`, and `action_model`.
    - If `keep_effect_transformer=True`:
      Retains the 2-layer `effect_transformer` self-attention on segment features (with
      `e_s = 0, e_r = 0`, no VLM/SceneGraph losses) to test whether unsupervised temporal
      self-attention inside segments helps.
    """
    pt_model = model.module if isinstance(model, nn.DataParallel) else model

    # Freeze unused GNN / VLM projection parameters in effect_model
    for name, param in pt_model.effect_model.named_parameters():
        if keep_effect_transformer and (
            "effect_transformer" in name or "enhanced_segment_proj" in name or "CLS_token" in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    def no_aem_effect_forward(feature_segments, video_list, segment_action_labels, contrast_mode, device):
        if not keep_effect_transformer:
            return {}, feature_segments
        # Optional: run only effect_transformer with zeroed state/relation tokens
        concat_labels = torch.cat(segment_action_labels, dim=0).to(device)
        mask = (concat_labels > 0).bool()
        fe = pt_model.effect_model.feature_enhancer
        enhanced_segments, _, _ = fe.enhance_feature(feature_segments, mask, device)
        return {}, enhanced_segments

    @torch.no_grad()
    def no_aem_effect_inference(feature_segments, action_labels, device):
        if not keep_effect_transformer:
            return feature_segments
        concat_labels = torch.cat(action_labels, dim=0).to(device)
        mask = (concat_labels > 0).bool()
        if not mask.any():
            return feature_segments
        fe = pt_model.effect_model.feature_enhancer
        enhanced_segments, _, _ = fe.enhance_feature(feature_segments, mask, device)
        return enhanced_segments

    pt_model.effect_model.forward = no_aem_effect_forward
    pt_model.effect_model.inference = no_aem_effect_inference
    return model


def build_split_test_loader(cfg, task_id, split_file, split_subset):
    """Build test dataloader matching `test_segmentation.py`."""
    root_dir = _resolve_root_dir(cfg)
    if split_file and cfg["dataset_name"] == "CaptainCook4D":
        allowed_vids = load_split_video_ids(split_file, subset=split_subset, task=str(task_id))
        test_dataset = CombinedCaptainCook4DDataset(
            root_dir=root_dir,
            task=str(task_id),
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
        test_loader = make_data_loader(test_dataset, False, None, 1, cfg["loader"]["num_workers"])
    return test_dataset, test_loader


@torch.no_grad()
def evaluate_model_on_loader(model, test_loader, clip_model, action_tokens):
    """Evaluate a model on `test_loader` and return segmentation + mistake detection metrics."""
    results = run_full_inference(model, test_loader, clip_model, action_tokens)
    gt_frames, pred_frames = [], []
    for _, v in results.items():
        gt_frames.append(v["frame_action_label"].clone().cpu().numpy())
        pred_frames.append(v["frame_action_pred"].clone().cpu().numpy())
    seg_w = compute_extended_seg_metrics(gt_frames, pred_frames, ignore_bg=False)
    seg_nb = compute_extended_seg_metrics(gt_frames, pred_frames, ignore_bg=True)
    ed = compute_ed_metrics(results)
    return {"with_bg": seg_w, "without_bg": seg_nb, "ed": ed}


def train_single_task(
    args,
    task_id: str,
    recipe_name: str,
    clip_model,
    activity_step_collection,
    step_id_mapping,
    split_file,
):
    """Train and evaluate the No-AEM model on a single CaptainCook4D recipe."""
    cfg = load_config(args.config)
    cfg["dataset"]["task"] = str(task_id)
    root_dir = _resolve_root_dir(cfg)
    device = cfg["devices"][0]

    if args.epochs > 0:
        cfg["opt"]["epochs"] = args.epochs
    if args.batch_size > 0:
        cfg["loader"]["batch_size"] = args.batch_size
    if args.lr > 0:
        cfg["opt"]["learning_rate"] = args.lr
    if args.prefix_length >= 0:
        cfg["model"]["action_model_cfg"]["prefix_length"] = args.prefix_length
    if args.ce_weight >= 0:
        cfg["model"]["action_model_cfg"]["ce_loss_weight"] = args.ce_weight
    if args.focal_weight >= 0:
        cfg["model"]["action_model_cfg"]["focal_loss_weight"] = args.focal_weight

    rng_generator = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    # 1. Build training dataset (normal error-free videos only, OCC setting)
    # Note: If split_file is provided and --respect_split_train is set, filter train videos by split_file[train]
    train_dataset = make_dataset(
        cfg["dataset_name"], True, None, cfg["train_split"], False, **cfg["dataset"]
    )
    if split_file and args.respect_split_train:
        train_allowed = load_split_video_ids(split_file, subset="train", task=str(task_id))
        train_dataset.annots = [a for a in train_dataset.annots if a["video_id"] in train_allowed]

    if len(train_dataset) == 0:
        print(f"  [Skip] Task {task_id} ({recipe_name}): 0 training videos.")
        return None

    # Use num_workers=0 so DataLoader shares the in-process float16 RAM cache without forking 4x RAM
    cfg["loader"]["num_workers"] = 0
    # Ensure batch_size does not exceed dataset size (drop_last=True would otherwise yield 0 batches!)
    effective_bs = min(cfg["loader"]["batch_size"], len(train_dataset))
    effective_bs = max(effective_bs, 1)
    if effective_bs != cfg["loader"]["batch_size"]:
        print(f"  [Notice] Task {task_id} has {len(train_dataset)} training videos; adjusting batch_size: {cfg['loader']['batch_size']} -> {effective_bs}")
    cfg["loader"]["batch_size"] = effective_bs
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg["loader"])
    if len(train_loader) == 0:
        # Fallback: force batch_size=1 so drop_last=True always yields at least len(train_dataset) batches
        cfg["loader"]["batch_size"] = 1
        train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg["loader"])

    test_dataset, test_loader = build_split_test_loader(cfg, task_id, split_file, args.split_subset)
    if len(test_dataset) == 0:
        print(f"  [Skip] Task {task_id} ({recipe_name}): 0 test videos.")
        return None

    # 2. Tokenize action prompts for this recipe
    action_tokens, num_classes = action_tokenize_ccp4d(
        clip_model, str(task_id), activity_step_collection, step_id_mapping, device=device
    )
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["backbone_type"] = "convTransformer"

    # 3. Create model & disable AEM (`Effect_Model`)
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = configure_no_aem_model(model, keep_effect_transformer=args.keep_effect_transformer)
    model = nn.DataParallel(model, device_ids=cfg["devices"]).to(device)

    cfg["opt"]["learning_rate"] *= len(cfg["devices"])
    optimizer = make_optimizer(model, cfg["opt"])
    scheduler = make_scheduler(optimizer, cfg["opt"], len(train_loader))
    model_ema = ModelEma(model)

    # 3b. Support auto-resuming interrupted runs from output_dir, --resume, or --init_from_ckpt
    max_epochs = cfg["opt"].get("early_stop_epochs", cfg["opt"]["epochs"] + cfg["opt"]["warmup_epochs"])
    os.makedirs(args.output_dir, exist_ok=True)
    safe_recipe = recipe_name.replace(" ", "")
    ckpt_name = f"recipe_{task_id}_{safe_recipe}_no_aem.pth.tar"
    saved_path = os.path.join(args.output_dir, ckpt_name)

    start_epoch = 0
    resume_ckpt = None
    is_same_task_resume = False

    # Priority 1: If output_dir already has a saved checkpoint for this exact task, auto-resume from it!
    if os.path.isfile(saved_path) and not args.no_auto_resume:
        resume_ckpt = saved_path
        is_same_task_resume = True
    elif args.resume:
        if os.path.isfile(args.resume):
            resume_ckpt = args.resume
            if os.path.abspath(args.resume) == os.path.abspath(saved_path):
                is_same_task_resume = True
        elif os.path.isdir(args.resume):
            pattern = os.path.join(args.resume, f"recipe_{task_id}_*.pth.tar")
            matches = sorted(glob.glob(pattern))
            if matches:
                resume_ckpt = matches[-1]
        if resume_ckpt is None:
            raise FileNotFoundError(f"=> no checkpoint found for task {task_id} at --resume='{args.resume}'")
    elif args.init_from_ckpt:
        pattern = os.path.join(args.paper_ckpt_dir, f"recipe_{task_id}_*.pth.tar")
        matches = sorted(glob.glob(pattern))
        if matches:
            resume_ckpt = matches[0]

    if resume_ckpt is not None:
        checkpoint = torch.load(resume_ckpt, map_location="cpu")

        def same_shape(state, target):
            # Exact logic from train.py: accept weights saved with or without 'module.' prefix;
            # skip effect_model.* and any tensors whose shape differs (e.g. classification head
            # when resuming per-recipe training from a pooled task='0' checkpoint).
            own = target.state_dict()
            matched, skipped = {}, []
            for k, v in state.items():
                while k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("effect_model.") and not args.keep_effect_transformer:
                    continue
                if isinstance(target, nn.DataParallel):
                    k = "module." + k
                if k in own and own[k].shape != v.shape:
                    skipped.append(k)
                elif k in own:
                    matched[k] = v
            return matched, skipped

        state, skipped = same_shape(checkpoint["state_dict"], model)
        model.load_state_dict(state, strict=False)
        ema_state = checkpoint.get("state_dict_ema", checkpoint["state_dict"])
        model_ema.module.load_state_dict(same_shape(ema_state, model_ema.module)[0], strict=False)

        # If resuming an interrupted run of the exact same task (no shape-mismatched head),
        # also restore epoch, optimizer, and scheduler (recomputing step if batch_size changed)!
        if (is_same_task_resume or (len(skipped) == 0 and checkpoint.get("task_id") == str(task_id))) and "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"])
            if "optimizer" in checkpoint and start_epoch < max_epochs:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception:
                    pass
            if start_epoch < max_epochs:
                # Re-build scheduler with `last_epoch = start_epoch * len(train_loader) - 1` so changing
                # --batch_size mid-training still aligns the cosine curve with `start_epoch`!
                scheduler = make_scheduler(
                    optimizer, cfg["opt"], len(train_loader), last_epoch=start_epoch * len(train_loader) - 1
                )
            print(
                f"  [Auto-Resume] Restored Task {task_id} from '{resume_ckpt}' at Epoch {start_epoch}/{max_epochs} "
                f"(batch_size={cfg['loader']['batch_size']}, {len(train_loader)} batches/ep)"
            )
        else:
            print(
                f"  [Warm-Start] Loaded {len(state)} tensors from '{resume_ckpt}'"
                + (f" (epoch {checkpoint['epoch']})" if "epoch" in checkpoint else "")
                + (f"; {len(skipped)} shape-mismatched tensors kept fresh init" if skipped else "")
            )
        del checkpoint

    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "-" * 115)
    print(
        f" Training Task {task_id:>2s}: {recipe_name} (No-AEM) | "
        f"Train vids: {len(train_dataset)} | Test vids: {len(test_dataset)} | "
        f"Epochs: {start_epoch}->{max_epochs} | Trainable Params: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M"
    )
    print("-" * 115)

    t0 = time.time()
    num_batches = len(train_loader)
    for epoch in range(start_epoch, max_epochs):
        ep_t0 = time.time()
        model.train()
        ep_losses = {"final": [], "cls": [], "reg": [], "det_ce": [], "det_focal": []}
        for b_idx, video_list in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            losses, _ = model(
                clip_model=clip_model,
                video_list=video_list,
                action_tokens=action_tokens,
                cur_epoch=epoch,
                tot_epoch=max_epochs,
            )
            losses["final_loss"].backward()
            if cfg["train_cfg"]["clip_grad_l2norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train_cfg"]["clip_grad_l2norm"])
            optimizer.step()
            scheduler.step()
            model_ema.update(model)

            ep_losses["final"].append(float(losses["final_loss"].item()))
            ep_losses["cls"].append(float(losses["cls_loss"].item()))
            ep_losses["reg"].append(float(losses["reg_loss"].item()))
            if "det_ce_loss" in losses:
                ep_losses["det_ce"].append(float(losses["det_ce_loss"].item()))
            if "det_focal_loss" in losses:
                ep_losses["det_focal"].append(float(losses["det_focal_loss"].item()))

            # Intra-epoch progress log for large pooled tasks (or when --batch_print_freq > 0)
            if args.batch_print_freq > 0 and (
                (b_idx + 1) % args.batch_print_freq == 0 or (b_idx + 1) == num_batches
            ):
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                print(
                    f"    [Ep {epoch+1:3d}/{max_epochs:3d} | Batch {b_idx+1:3d}/{num_batches:3d}] "
                    f"loss: {ep_losses['final'][-1]:.3f} (avg: {np.mean(ep_losses['final']):.3f}) | "
                    f"VRAM: {vram_gb:.2f}GB | {time.time() - ep_t0:.1f}s",
                    flush=True,
                )

        if (epoch + 1) % args.print_freq == 0 or (epoch + 1) == max_epochs or epoch == start_epoch:
            lr_now = scheduler.get_last_lr()[0]
            ep_sec = time.time() - ep_t0
            done_eps = (epoch + 1) - start_epoch
            total_sec = time.time() - t0
            eta_sec = (total_sec / max(done_eps, 1)) * (max_epochs - (epoch + 1))
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            loss_final_str = f"{np.mean(ep_losses['final']):.3f}" if ep_losses["final"] else "0.000"
            loss_cls_str = f"{np.mean(ep_losses['cls']):.3f}" if ep_losses["cls"] else "0.000"
            loss_reg_str = f"{np.mean(ep_losses['reg']):.3f}" if ep_losses["reg"] else "0.000"
            print(
                f"  Epoch [{epoch+1:3d}/{max_epochs:3d}] | lr: {lr_now:.2e} | "
                f"loss: {loss_final_str} "
                f"(cls: {loss_cls_str}, reg: {loss_reg_str}, "
                f"det_ce: {np.mean(ep_losses['det_ce'] or [0]):.3f}, "
                f"det_focal: {np.mean(ep_losses['det_focal'] or [0]):.3f}) | "
                f"{ep_sec:.1f}s/ep (ETA: {eta_sec/60:.1f}m) | PeakVRAM: {vram_gb:.1f}GB",
                flush=True,
            )

        # Periodically save checkpoint every --ckpt_freq epochs so Colab disconnects lose nothing
        if (args.ckpt_freq > 0 and (epoch + 1) % args.ckpt_freq == 0) or (epoch + 1) == max_epochs:
            eval_model = model_ema.module if args.eval_ema else model
            save_state = {
                "task_id": str(task_id),
                "epoch": epoch + 1,
                "state_dict": eval_model.state_dict(),
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(save_state, False, file_folder=args.output_dir, file_name=ckpt_name)

    train_time = time.time() - t0
    eval_model = model_ema.module if args.eval_ema else model
    configure_no_aem_model(eval_model, keep_effect_transformer=args.keep_effect_transformer)

    # 4. Evaluate our newly trained No-AEM model
    our_metrics = evaluate_model_on_loader(eval_model, test_loader, clip_model, action_tokens)
    w = our_metrics["with_bg"]
    ed = our_metrics["ed"]
    print(
        f"  => [Trained No-AEM]  Acc: {w['Accuracy']:5.2f}% | Edit: {w['Edit']:5.2f}% | F1@50: {w['F1@50']:5.2f}% | "
        f"Frag: {w['frag_ratio']:4.2f}x || Prec: {ed['precision']:5.2f}% | AUC: {ed['auc']:5.2f}% | EDA: {ed['eda']:5.2f}% "
        f"(Trained in {train_time:.1f}s -> {saved_path})"
    )

    # 5. Also evaluate the paper's released Full-AEM checkpoint on the exact same test_loader for direct comparison
    paper_metrics = None
    pattern = os.path.join(args.paper_ckpt_dir, f"recipe_{task_id}_*.pth.tar")
    matches = sorted(glob.glob(pattern))
    if matches:
        paper_model = make_meta_arch(cfg["model_name"], **cfg["model"])
        paper_model = nn.DataParallel(paper_model, device_ids=cfg["devices"]).to(device)
        paper_model = load_weights(paper_model, matches[0], use_ema=False)
        paper_metrics = evaluate_model_on_loader(paper_model, test_loader, clip_model, action_tokens)
        pw = paper_metrics["with_bg"]
        ped = paper_metrics["ed"]
        print(
            f"  => [Paper Full-AEM]  Acc: {pw['Accuracy']:5.2f}% | Edit: {pw['Edit']:5.2f}% | F1@50: {pw['F1@50']:5.2f}% | "
            f"Frag: {pw['frag_ratio']:4.2f}x || Prec: {ped['precision']:5.2f}% | AUC: {ped['auc']:5.2f}% | EDA: {ped['eda']:5.2f}%"
        )
        del paper_model

    del model, model_ema, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "task_id": str(task_id),
        "recipe": recipe_name,
        "train_time": train_time,
        "ours": our_metrics,
        "paper": paper_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train & Evaluate the Streamlined No-AEM Model on CaptainCook4D."
    )
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "ccp4d.yaml"),
                        help="Path to CaptainCook4D config YAML")
    parser.add_argument("--tasks", type=str, default="1",
                        help="Recipe ID(s) to train: '0' or 'pooled' for all recipes pooled together (351 classes), "
                             "'all' for all 24 recipes trained sequentially, or e.g. '1,10,12'")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs per recipe (paper default: 100)")
    parser.add_argument("--batch_size", type=int, default=-1,
                        help="Override training batch_size (default: use config's 2; e.g. --batch_size 8 or 16)")
    parser.add_argument("--lr", type=float, default=-1.0,
                        help="Override learning rate (default: use config's 0.0001)")
    parser.add_argument("--prefix_length", type=int, default=-1,
                        help="Override prompt prefix length M (e.g. 4 for learnable prefix, 0 for frozen CLIP text only)")
    parser.add_argument("--ce_weight", type=float, default=-1.0,
                        help="Override Action_Model Cross-Entropy loss weight (default: 1.0)")
    parser.add_argument("--focal_weight", type=float, default=-1.0,
                        help="Override Action_Model Focal loss weight (default: 1.0)")
    parser.add_argument("--keep_effect_transformer", action="store_true",
                        help="Keep the 2-layer self-attention transformer on segments (still without VLM/Effect supervision)")
    parser.add_argument("--eval_ema", action="store_true",
                        help="Evaluate using EMA weights instead of direct model weights")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to a checkpoint (.pth.tar) or checkpoint folder to resume/warm-start from "
                             "(uses train.py's same_shape logic, so you can pass a pooled task='0' checkpoint)")
    parser.add_argument("--init_from_ckpt", action="store_true",
                        help="Shortcut to warm-start each recipe from its corresponding paper checkpoint in --paper_ckpt_dir")
    parser.add_argument("--split_file", type=str,
                        default=os.path.join(TESTS_DIR, "person_data_split_combined.json"),
                        help="Path to split JSON for evaluation, or 'none' for paper's default error-only test split")
    parser.add_argument("--split_subset", type=str, default="test")
    parser.add_argument("--respect_split_train", action="store_true",
                        help="Also filter training normal videos by split_file['train'] (by default uses all normal videos as in paper)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ccp4d_no_aem"),
                        help="Directory to save trained No-AEM checkpoints")
    parser.add_argument("--paper_ckpt_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ccp4d"),
                        help="Directory containing the paper's released checkpoints for side-by-side comparison")
    parser.add_argument("--print_freq", type=int, default=5,
                        help="Print epoch summary every N epochs (default: 5)")
    parser.add_argument("--batch_print_freq", type=int, default=15,
                        help="Print intra-epoch progress every N batches (default: 15, set 0 to disable)")
    parser.add_argument("--ckpt_freq", type=int, default=5,
                        help="Save checkpoint to --output_dir every N epochs for seamless auto-resume (default: 5)")
    parser.add_argument("--no_auto_resume", action="store_true",
                        help="Start from scratch even if a checkpoint already exists in --output_dir")
    args = parser.parse_args()

    split_file = None if (args.split_file and args.split_file.lower() == "none") else args.split_file
    cfg_base = load_config(args.config)
    root_dir = _resolve_root_dir(cfg_base)
    device = cfg_base["devices"][0]

    with open(os.path.join(root_dir, "id_activity_mapping.json"), "r") as fp:
        id_activity = json.load(fp)
    id_activity["0"] = "AllRecipesPooled"
    with open(os.path.join(root_dir, "activity_step_collection.json"), "r") as fp:
        activity_step_collection = json.load(fp)
    with open(os.path.join(root_dir, "step_id_mapping.json"), "r") as fp:
        step_id_mapping = json.load(fp)

    tasks_arg = args.tasks.strip().lower()
    if tasks_arg == "all":
        task_list = [(str(tid), rname) for tid, rname in id_activity.items() if str(tid) != "0"]
    elif tasks_arg in ("0", "pooled", "joint"):
        task_list = [("0", id_activity["0"])]
    else:
        wanted = ["0" if t.strip().lower() in ("pooled", "joint") else t.strip()
                  for t in args.tasks.split(",") if t.strip()]
        task_list = [(t, id_activity[t]) for t in wanted if t in id_activity]

    if not task_list:
        raise ValueError(f"No valid CaptainCook4D tasks matched --tasks={args.tasks}")

    print("=" * 125)
    print(
        f" TRAINING STREAMLINED NO-AEM BASELINE ON CAPTAINCOOK4D | "
        f"Tasks: {len(task_list)} | Epochs: {args.epochs} | Output: {args.output_dir}"
    )
    print(" Active Components : ActionFormer + Dynamic Fusion Head + EVA02-CLIP Prompt Detector (CE + Focal Loss)")
    print(" Removed Components: Effect Frame Sampling, GroundingDINO, GPT-4o Scene Graphs, GAT, Effect Tokens (e_s, e_r)")
    print("=" * 125)

    clip_model = load_clip_model(device)

    all_runs = []
    for tid, rname in task_list:
        res = train_single_task(
            args, tid, rname, clip_model, activity_step_collection, step_id_mapping, split_file
        )
        if res is not None:
            all_runs.append(res)

    if not all_runs:
        return

    # Print final side-by-side comparison table across all trained recipes
    print("\n" + "=" * 136)
    print(f" FINAL COMPARISON: NEWLY TRAINED NO-AEM vs. PAPER RELEASED FULL-AEM ({len(all_runs)} RECIPES)")
    print("=" * 136)
    print(
        f"{'Task':<5} {'Recipe':<30} | {'Model Variant':<16} | "
        f"{'Acc(w/BG)':>9} {'Edit':>6} {'F1@50':>6} {'Frag':>6} | "
        f"{'Prec':>6} {'AUC':>6} {'EDA':>6} | {'Time':>6}"
    )
    print("-" * 136)

    our_w_list, our_ed_list = [], []
    pap_w_list, pap_ed_list = [], []

    for r in all_runs:
        tid = r["task_id"]
        rname = r["recipe"][:30]
        ow = r["ours"]["with_bg"]
        oed = r["ours"]["ed"]
        our_w_list.append(ow)
        our_ed_list.append(oed)

        print(
            f"{tid:<5} {rname:<30} | {'Trained No-AEM':<16} | "
            f"{ow['Accuracy']:>8.2f}% {ow['Edit']:>6.2f} {ow['F1@50']:>6.2f} {ow['frag_ratio']:>5.2f}x | "
            f"{oed['precision']:>6.2f} {oed['auc']:>6.2f} {oed['eda']:>6.2f} | {r['train_time']:>5.1f}s"
        )
        if r["paper"] is not None:
            pw = r["paper"]["with_bg"]
            ped = r["paper"]["ed"]
            pap_w_list.append(pw)
            pap_ed_list.append(ped)
            print(
                f"{'':<5} {'':<30} | {'Paper Full-AEM':<16} | "
                f"{pw['Accuracy']:>8.2f}% {pw['Edit']:>6.2f} {pw['F1@50']:>6.2f} {pw['frag_ratio']:>5.2f}x | "
                f"{ped['precision']:>6.2f} {ped['auc']:>6.2f} {ped['eda']:>6.2f} | {'--':>6}"
            )
        print("-" * 136)

    # Print macro averages
    def _avg_row(label, w_list, ed_list):
        acc = np.mean([x["Accuracy"] for x in w_list])
        edit = np.mean([x["Edit"] for x in w_list])
        f1 = np.mean([x["F1@50"] for x in w_list])
        frag = sum(x["pred_segs"] for x in w_list) / max(1, sum(x["gt_segs"] for x in w_list))
        prec = np.mean([x["precision"] for x in ed_list])
        auc = np.mean([x["auc"] for x in ed_list])
        eda = np.mean([x["eda"] for x in ed_list])
        print(
            f"{'AVG':<5} {f'({len(w_list)} recipes)':<30} | {label:<16} | "
            f"{acc:>8.2f}% {edit:>6.2f} {f1:>6.2f} {frag:>5.2f}x | "
            f"{prec:>6.2f} {auc:>6.2f} {eda:>6.2f} |"
        )

    _avg_row("Trained No-AEM", our_w_list, our_ed_list)
    if pap_w_list:
        _avg_row("Paper Full-AEM", pap_w_list, pap_ed_list)
    print("=" * 136)


if __name__ == "__main__":
    main()
