import argparse
import os
import time
import datetime
import json
import warnings
import torch
import torch.nn as nn
import open_clip

from tensorboardX import SummaryWriter
from pprint import pprint
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, save_checkpoint,
                        make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from libs.datasets.clip_encode import action_tokenize_egoper
from libs.datasets.clip_encode_ccp4d import action_tokenize_ccp4d

warnings.filterwarnings("ignore")

def checkpoint_states(epoch, model, model_ema, optimizer, scheduler):
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'state_dict_ema': model_ema.module.state_dict()
    }


def train_model(
    cfg, args,
    model, model_ema,
    optimizer, scheduler,
    train_loader, val_loader,
    action_tokens, clip_model,
    tb_writer, ckpt_folder
):

    print("\n--- Starting Training ---")
    max_epochs = cfg['opt'].get('early_stop_epochs', cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])
    best_val_loss = float('inf')

    for epoch in range(max_epochs):
        model_ema = train_one_epoch(
            train_loader, action_tokens, model, clip_model, optimizer, scheduler, epoch, max_epochs,
            model_ema=model_ema, clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'], tb_writer=tb_writer
        )

        # Keep the epoch with the lowest validation loss
        if val_loader is not None:
            val_losses = valid_one_epoch(
                val_loader, action_tokens, model, clip_model, epoch, max_epochs, tb_writer=tb_writer
            )
            if val_losses['final_loss'] < best_val_loss:
                best_val_loss = val_losses['final_loss']
                save_checkpoint(checkpoint_states(epoch + 1, model, model_ema, optimizer, scheduler), False,
                                file_folder=ckpt_folder, file_name='best_val.pth.tar')

        # Save checkpoint periodically
        if ((epoch + 1) == max_epochs) or ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)):
            save_checkpoint(
                checkpoint_states(epoch + 1, model, model_ema, optimizer, scheduler), False,
                file_folder=ckpt_folder,
                file_name=f'epoch_{epoch + 1:03d}.pth.tar'
            )


def main(args):
    """
    Main function to orchestrate the setup, training, and evaluation.
    """

    """1. Setup parameters / folders"""
    args.start_epoch = 0
    cfg = load_config(args.config)
    if args.task is not None:
        cfg['dataset']['task'] = args.task

    # Prepare output folder
    os.makedirs(cfg['output_folder'], exist_ok=True)
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(args.output))
    os.makedirs(ckpt_folder, exist_ok=True)
        
    logger_folder = os.path.join(ckpt_folder, 'logs', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tb_writer = SummaryWriter(logger_folder)

    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # Scale learning rate and workers based on GPU count
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    # Load CLIP model
    clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k",
        device="cuda"
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    # CLIP stays frozen; only the prompt prefix fed through its text tower is learned
    clip_model = (clip.eval().requires_grad_(False), tokenizer, preprocess)

    """2. Create datasets / dataloaders"""
    # Training set
    train_dataset = make_dataset(cfg['dataset_name'], True, clip_model, cfg['train_split'], args.use_gcn, **cfg['dataset'])
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])

    # Validation set.
    val_loader = None
    if cfg['val_split']:
        val_dataset = make_dataset(cfg['dataset_name'], False, clip_model, cfg['val_split'], args.use_gcn, **cfg['dataset'])
        val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])

    # Load annotations and tokenize actions
    if cfg['dataset_name'] == 'CaptainCook4D':
        with open(os.path.join(cfg["dataset"]["root_dir"], "activity_step_collection.json"), 'r') as fp:
            activity_step_collection = json.load(fp)
        with open(os.path.join(cfg["dataset"]["root_dir"], "step_id_mapping.json"), 'r') as fp:
            step_id_mapping = json.load(fp)
        action_tokens, num_classes = action_tokenize_ccp4d(
            clip_model, cfg['dataset']['task'],
            activity_step_collection, step_id_mapping, device="cuda"
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
            device="cuda"
        )

    """3. Create model, optimizer, and scheduler"""
    cfg['model']['backbone_type'] = 'convGCNTransformer' if args.use_gcn else 'convTransformer'
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    print("Using model EMA ...")
    model_ema = ModelEma(model)

    # Resume from a checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            # load on CPU; load_state_dict copies into the model's device
            checkpoint = torch.load(args.resume, map_location='cpu')

            def same_shape(state, target):
                # accept weights saved with or without the DataParallel 'module.' prefix (released
                # checkpoints have none); tensors whose shape differs (e.g. the classification head
                # of a checkpoint trained with another set of classes, such as the pooled
                # CaptainCook4D task '0') keep their fresh initialisation
                own = target.state_dict()
                matched, skipped = {}, []
                for k, v in state.items():
                    while k.startswith('module.'):
                        k = k[len('module.'):]
                    k = 'module.' + k
                    if k in own and own[k].shape != v.shape:
                        skipped.append(k)
                    elif k in own:
                        matched[k] = v
                return matched, skipped

            state, skipped = same_shape(checkpoint['state_dict'], model)
            model.load_state_dict(state, strict=False)
            # released checkpoints carry no EMA weights: start the EMA from the loaded model
            ema_state = checkpoint.get('state_dict_ema', checkpoint['state_dict'])
            model_ema.module.load_state_dict(same_shape(ema_state, model_ema.module)[0], strict=False)
            print(f"=> loaded {len(state)} tensors from '{args.resume}'"
                  + (f" (epoch {checkpoint['epoch']})" if 'epoch' in checkpoint else "")
                  + (f"; {len(skipped)} tensors with another shape keep their initialisation: {skipped}" if skipped else ""))
            del checkpoint
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return

    # Save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()
    
    """4. Run Training and Evaluation"""
    # Call the dedicated training function
    train_model(
        cfg, args, model, model_ema, optimizer, scheduler, train_loader, val_loader,
        action_tokens, clip_model, tb_writer, ckpt_folder
    )
    
    tb_writer.close()
    print("\nAll done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Evaluate a transformer for action localization')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int, help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str, help='name of experiment folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint to resume training (default: none)')
    parser.add_argument('--task', type=str, default=None, help='override dataset.task in the config (e.g. a CaptainCook4D recipe id)')
    parser.add_argument('--use_gcn', default=False, action='store_true', help='use GCN in the model backbone')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions for evaluation (default: -1)')
    
    args = parser.parse_args()
    main(args)