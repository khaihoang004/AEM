import argparse
import os
import glob
import time
import json
import warnings
from collections import OrderedDict

import open_clip
import torch
import torch.nn as nn
import numpy as np

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from metric import calculate_micro_auc, evaluate_ed, evaluate_seg
from libs.datasets.clip_encode import action_tokenize_egoper
from libs.datasets.clip_encode_ccp4d import action_tokenize_ccp4d

warnings.filterwarnings("ignore")

# Detection thresholds used for EDA / Precision / AUC. Following EgoPER (Lee et al., 2024),
# the detection results are aggregated over a dense threshold grid instead of a single
# operating point: 40 thresholds, 0.000, 0.025, ..., 0.975.
THRESHOLDS = np.arange(0.0, 1.0, 0.025)


def load_clip_model(device="cuda"):
    """Load the frozen EVA-02 CLIP used for prompt / effect-frame encoding."""
    clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k",
        device=device
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    return (clip.eval().requires_grad_(False).to(device), tokenizer, preprocess)


def resolve_ckpt_file(ckpt, epoch=-1, is_best=False):
    """Resolve a checkpoint path, which may be a file or a folder of epoch checkpoints."""
    if ".pth.tar" in ckpt:
        assert os.path.isfile(ckpt), "CKPT file does not exist!"
        return ckpt
    assert os.path.isdir(ckpt), "CKPT file folder does not exist!"
    if epoch > 0:
        ckpt_file = os.path.join(ckpt, 'epoch_{:03d}.pth.tar'.format(epoch))
    elif is_best:
        # lowest validation loss, written by train.py when the config has a validation split
        ckpt_file = os.path.join(ckpt, 'best_val.pth.tar')
    else:
        ckpt_file = sorted(glob.glob(os.path.join(ckpt, '*.pth.tar')))[-1]
    assert os.path.exists(ckpt_file)
    return ckpt_file


def load_weights(model, ckpt_file, use_ema=False):
    """Load checkpoint weights into `model` (an nn.DataParallel wrapper).

    The released checkpoints store the *unwrapped* module weights (no `module.` prefix),
    while checkpoints written from a DataParallel-wrapped model carry the prefix. Both are
    accepted here: the prefix is normalized to match the model that is being loaded into.
    """
    # load on CPU; load_state_dict copies into the model's device
    checkpoint = torch.load(ckpt_file, map_location='cpu')

    if use_ema:
        assert 'state_dict_ema' in checkpoint, \
            "--use_ema requested but this checkpoint has no 'state_dict_ema' entry."
        state_dict, source = checkpoint['state_dict_ema'], 'EMA weights'
    else:
        state_dict, source = checkpoint['state_dict'], 'model weights'

    wrapped = isinstance(model, nn.DataParallel)
    normalized = OrderedDict()
    for k, v in state_dict.items():
        while k.startswith('module.'):
            k = k[len('module.'):]
        normalized['module.' + k if wrapped else k] = v

    model.load_state_dict(normalized)
    print("=> loaded {} from '{}' ({} tensors)".format(source, ckpt_file, len(normalized)))
    del checkpoint
    return model


def compute_metrics(results):
    """Aggregate a single task's inference results into the reported metrics.

    Returns Precision / AUC / EDA for one task (one EgoPER task or one CaptainCook4D recipe):
      - EDA:       segment-level detection accuracy, averaged over the 40 thresholds.
      - Precision: frame-level precision of the error class (TP / (TP + FP), error = positive),
                   pooled over all test videos of the task, averaged over the same 40 thresholds.
      - AUC:       area under the micro (frame-level) ROC traced by those 40 thresholds.
    """
    eda_list, precision_list, micro_fprs, micro_tprs = [], [], [], []
    for threshold in THRESHOLDS:
        outputs = evaluate_ed(results, threshold=float(threshold))
        eda_list.append(outputs['error_acc'])
        precision_list.append(outputs['micro']['precision'])
        micro_fprs.append(outputs['micro']['fpr'])
        micro_tprs.append(outputs['micro']['tpr'])

    return {
        'eda': float(np.mean(eda_list)) * 100,
        'precision': float(np.mean(precision_list)) * 100,
        'auc': calculate_micro_auc(micro_fprs, micro_tprs) * 100,
    }


def run_inference(model, test_loader, clip_model, action_tokens):
    """Segment and score every video of `test_loader` with `model` (an nn.DataParallel wrapper).

    Returns the per-video results dict consumed by `compute_metrics` / `evaluate_seg`.
    """
    model.eval()
    results = {}
    for video_list in test_loader:
        with torch.no_grad():
            results.update(model.module.inference(
                video_list=video_list,
                clip_model=clip_model,
                action_tokens=action_tokens
            ))
    return results


def evaluate(cfg, ckpt_file, clip_model, use_gcn=False, use_ema=False):
    """Evaluate one checkpoint on one task; returns the segmentation and detection metrics."""
    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    test_dataset = make_dataset(
            cfg['dataset_name'], False, clip_model, cfg['test_split'], use_gcn, **cfg['dataset']
        )
    # set bs = 1, and disable shuffle
    test_loader = make_data_loader(
        test_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    ## move clip model to device
    clip_model = (clip_model[0].to(cfg['devices'][0]), clip_model[1], clip_model[2])

    # tokenize the recipe's / task's action prompts
    if cfg['dataset_name'] == 'CaptainCook4D':
        with open(os.path.join(cfg["dataset"]["root_dir"], "activity_step_collection.json"), 'r') as fp:
            activity_step_collection = json.load(fp)
        with open(os.path.join(cfg["dataset"]["root_dir"], "step_id_mapping.json"), 'r') as fp:
            step_id_mapping = json.load(fp)
        action_tokens, num_classes = action_tokenize_ccp4d(
            clip_model, cfg['dataset']['task'],
            activity_step_collection, step_id_mapping, device=cfg['devices'][0]
        )
        cfg['model']['num_classes'] = num_classes
    else:
        with open(os.path.join(cfg["dataset"]["root_dir"], "annotation.json"), 'r') as fp:
            all_annot = json.load(fp)
        task_annot = all_annot[cfg['dataset']['task']]
        action_tokens = action_tokenize_egoper(
            clip_model,
            cfg['dataset']['task'],
            annot=task_annot,
            device=cfg['devices'][0]
        )

    """3. create model and evaluator"""
    # model
    cfg['model']['backbone_type'] = 'convGCNTransformer' if use_gcn else 'convTransformer'
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    model = load_weights(model, ckpt_file, use_ema=use_ema)

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    results = run_inference(model, test_loader, clip_model, action_tokens)

    """6. Segmentation Accuracy"""
    list_of_gt_frame_action = []
    list_of_pred_frame_action = []
    for k,v in results.items():
        list_of_gt_frame_action.append(v['frame_action_label'].clone().cpu().numpy())
        list_of_pred_frame_action.append(v['frame_action_pred'].clone().cpu().numpy())    
    seg_outputs = evaluate_seg(list_of_gt_frame_action, list_of_pred_frame_action)
    print("Segmentation Accuracy: {:.3f} mIoU: {:.3f} Edit: {:.3f} F1@0.5:{:.3f}".format(\
        seg_outputs['Accuracy'], seg_outputs['mIoU'], seg_outputs['Edit'], seg_outputs['F1@0.5'])
    )

    """7. Mistake detection: Precision / AUC / EDA over the threshold grid"""
    metrics = compute_metrics(results)
    end = time.time()

    print("Detection Error Accuracy: {:0.2f}%, Precision: {:0.2f}%, Micro AUC: {:.2f}%".format(
        metrics['eda'], metrics['precision'], metrics['auc'])
    )
    print("All done! Total time: {:0.2f} sec".format(end - start))

    metrics.update(seg_outputs)
    return metrics


def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    ckpt_file = resolve_ckpt_file(args.ckpt, args.epoch, args.is_best)

    if args.task is not None:
        cfg['dataset']['task'] = args.task
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    clip_model = load_clip_model(cfg['devices'][0])
    evaluate(cfg, ckpt_file, clip_model, use_gcn=args.use_gcn, use_ema=args.use_ema)
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Evaluate a trained mistake-detection model on one task / recipe')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--is_best', action='store_true')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--task', type=str, default=None,
                help="override dataset.task in the config (e.g. a CaptainCook4D recipe id)")
    parser.add_argument('--use_gcn', default=False, action='store_true',
                help='use the GCN backbone (active-object graph); required for EgoPER')
    parser.add_argument('--use_ema', default=False, action='store_true',
                help="evaluate the EMA weights ('state_dict_ema') instead of 'state_dict'")
    args = parser.parse_args()
    main(args)
