import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from copy import deepcopy
from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config['weight_decay']
    )

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _log_epoch(split, loss_meters, acc_meters, curr_epoch, tb_writer, scheduler=None):
    """Write the epoch averages of every loss / accuracy to tensorboard and stdout."""
    if tb_writer is not None:
        for name, meter in {**loss_meters, **acc_meters}.items():
            tb_writer.add_scalar(f'{split}/{name}', meter.avg, curr_epoch)
        if scheduler is not None:
            tb_writer.add_scalar(f'{split}/lr', scheduler.get_last_lr()[0], curr_epoch)

    prefix = f"[{split.capitalize()}]: Epoch {curr_epoch}"
    loss_print = ", ".join(f"{name}={meter.avg:.3f}" for name, meter in loss_meters.items())
    acc_print = ", ".join(f"{name}={meter.avg:.3f}" for name, meter in acc_meters.items())
    print(f"{prefix}\n{loss_print}\n{acc_print}")


def _update_meters(loss_meters, acc_meters, losses, topk_acc):
    for name, value in losses.items():
        loss_meters.setdefault(name, AverageMeter()).update(float(value))
    for name, value in topk_acc.items():
        acc_meters.setdefault(name, AverageMeter()).update(float(value))


def train_one_epoch(
    train_loader,
    action_tokens,
    model,
    clip_model,
    optimizer,
    scheduler,
    curr_epoch,
    tot_epoch,
    model_ema = None,
    clip_grad_l2norm = 1.0,
    tb_writer = None
):
    """Training the model for one epoch"""
    loss_meters, acc_meters = {}, {}

    model.train()
    for iter_idx, video_list in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)
        losses, topk_acc = model(
            clip_model=clip_model,
            video_list=video_list,
            action_tokens=action_tokens,
            cur_epoch=curr_epoch,
            tot_epoch=tot_epoch
        )
        losses['final_loss'].backward()
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        _update_meters(loss_meters, acc_meters, losses, topk_acc)

    _log_epoch('train', loss_meters, acc_meters, curr_epoch, tb_writer, scheduler)
    return model_ema


def valid_one_epoch(
    val_loader,
    action_tokens,
    model,
    clip_model,
    curr_epoch,
    tot_epoch,
    tb_writer = None
):
    """Validate the model for one epoch and return the average of every loss.

    Runs under no_grad and in eval mode, so it neither builds a graph nor moves the
    model's running loss normalizer (see PtTransformer._compute_losses).
    """
    loss_meters, acc_meters = {}, {}

    model.eval()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            losses, topk_acc = model(
                clip_model=clip_model,
                video_list=video_list,
                action_tokens=action_tokens,
                cur_epoch=curr_epoch,
                tot_epoch=tot_epoch
            )

        torch.cuda.synchronize()
        _update_meters(loss_meters, acc_meters, losses, topk_acc)

    _log_epoch('val', loss_meters, acc_meters, curr_epoch, tb_writer)
    return {name: meter.avg for name, meter in loss_meters.items()}
