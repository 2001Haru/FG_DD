import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
try:
    import wandb
except ImportError:
    wandb = None
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils_validate import AverageMeter, accuracy, get_parameters, load_val_loader, load_small_dataset_model
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from relabel.utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX,
                               RandomHorizontalFlipWithRes,
                               RandomResizedCropWithCoords, mix_aug)

_original_map_dataset_fetch = _MapDatasetFetcher.fetch


def _cvdd_fkd_map_dataset_fetch(self, possibly_batched_index):
    """Repository-local equivalent of CV-DD's documented PyTorch patch."""
    if not (hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load'):
        return _original_map_dataset_fetch(self, possibly_batched_index)

    mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(
        possibly_batched_index[0]
    )
    if self.auto_collation:
        if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
            data = self.dataset.__getitems__(possibly_batched_index)
        else:
            data = [self.dataset[idx] for idx in possibly_batched_index]
    else:
        data = self.dataset[possibly_batched_index]
    return self.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()


if not getattr(_MapDatasetFetcher.fetch, "_cvdd_fkd_patch", False):
    _cvdd_fkd_map_dataset_fetch._cvdd_fkd_patch = True
    _MapDatasetFetcher.fetch = _cvdd_fkd_map_dataset_fetch


def get_args():
    parser = argparse.ArgumentParser("FKD Training on Cifar-100")
    parser.add_argument('--exp-name', type=str,
                        default="", help='the name of the run')
    parser.add_argument('--original-data-path', required='True', type=str,
                        help='name of the original data')
    parser.add_argument('--simple', default=False,action='store_true',)
    parser.add_argument('--fkd-path', default=None, type=str,
                        help='path to the fkd labels')
    parser.add_argument('--hard-label', action='store_true',
                        help='train directly from ImageFolder class IDs with cross-entropy; do not load FKD labels')
    parser.add_argument('--output-dir', required='True', type=str,
                        help='output directory')
    parser.add_argument('--dataset-name',default='cifar100',type=str,
                        help='dataset name')
    parser.add_argument('--min-scale', type=float, default=0.08, )
    parser.add_argument('--batch-size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='reuse DataLoader workers across epochs (off matches released CV-DD)')
    parser.add_argument('--train-seed', type=int, default=None,
                        help='optional seed for student initialization and runtime randomness')
    parser.add_argument('--ipc',type=int,help='number of images per class')
    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--sgd-lr', type=float,
                        default=0.01, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.5, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    parser.add_argument('--adamw-lr-override', type=float, default=None,
                        help='override the dataset/model-specific AdamW learning rate')
    parser.add_argument('--eta-override', type=float, default=None,
                        help='override the dataset/model-specific cosine eta')
    parser.add_argument('--model', type=str,
                        default='ResNet18', help='student model name')
    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--wandb-project', type=str,
                        default='RankDD', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--disable-wandb', action='store_true',
                        help='disable external logging for local/offline runs')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--val-dir', required=True, type=str,
                        help="path to the validation data")
    parser.add_argument('--per-class-output', type=str, default=None,
                        help='write best-checkpoint per-class validation accuracy as JSON')
    parser.add_argument('--eval-hierarchy-mapping', type=str, default=None,
                        help='fine_to_coarse hierarchy for collapsed probability evaluation')
    parser.add_argument('--primary-eval-collapsed-coarse', action='store_true',
                        help='select best checkpoint using hierarchy-collapsed coarse Top1')

    args = parser.parse_args()

    if not args.hard_label and args.fkd_path is None:
        parser.error('--fkd-path is required unless --hard-label is specified')
    if args.hard_label and args.mix_type is not None:
        parser.error('--hard-label does not use FKD MixUp/CutMix; omit --mix-type')

    args.mode = 'fkd_load'

    # final checked
    if args.dataset_name == 'cifar10':
        args.mean_norm = [0.4914, 0.4822, 0.4465]
        args.std_norm = [0.2470, 0.2435, 0.2616]
        args.ncls = 10
        args.input_size = 32
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            if args.epochs == 1000:
                args.adamw_lr = 0.0005
                args.eta = 1
            else:
                args.adamw_lr = 0.001
                args.eta = 1
            
        else:
            args.adamw_lr = 0.0005
            args.eta = 1
    
    # final checked
    elif args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.input_size = 32
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 0.001
            if args.ipc==10:
                args.eta=2
            else:
                args.eta = 1
        else:
            args.adamw_lr = 0.0005
            args.eta = 1

    elif args.dataset_name == 'cifar20':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 20
        args.input_size = 32
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 0.001
            args.eta = 2 if args.ipc == 10 else 1
        else:
            args.adamw_lr = 0.0005
            args.eta = 1
    
    #final checked
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.input_size = 64
        
        if args.model == 'ResNet18':
            args.adamw_lr = 0.001
            if args.ipc ==10:
                args.eta = 1
            else:
                args.eta = 2
            
        elif args.model == 'ResNet50':
            args.adamw_lr = 0.001
            args.eta = 1

        else:
            args.adamw_lr = 0.0005
            args.eta = 2
    
    # final checked
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.input_size = 224
        if args.model == 'ResNet18':
            args.adamw_lr = 0.0005
            if args.ipc == 10:
                args.eta = 1
            else:
                args.eta = 2
        elif args.model == 'ResNet50':
            # lr
            if args.ipc == 10:
                args.adamw_lr = 0.001
            else:
                args.adamw_lr = 0.0005
            # eta
            if args.ipc == 1:
                args.eta = 2
            else:
                args.eta = 1
        else:
            # lr
            if args.ipc == 50:
                args.adamw_lr = 0.001
            else:
                args.adamw_lr = 0.0005

            args.eta = 2
    
    # final checked
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.input_size = 224
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            # lr
            args.adamw_lr = 0.001
            if args.ipc == 50:
                args.eta = 1
            else:
                args.eta = 2
        elif args.model == 'ResNet101':
            # lr
            if args.ipc == 10 or args.ipc == 1:
                args.adamw_lr = 0.001
            elif args.ipc == 50:
                args.adamw_lr = 0.0005
            args.eta = 2
    elif args.dataset_name == 'CUB_imsize224':
        args.mean_norm = [0.4857, 0.4994, 0.4326]
        args.std_norm = [0.2260, 0.2215, 0.2595]
        args.ncls = 200
        args.input_size = 224
        args.adamw_lr = 0.001
        args.eta = 2
    else:
        raise ValueError('dataset not supported')

    if args.adamw_lr_override is not None:
        if args.adamw_lr_override <= 0:
            raise ValueError('--adamw-lr-override must be positive')
        args.adamw_lr = args.adamw_lr_override
    if args.eta_override is not None:
        if args.eta_override <= 0:
            raise ValueError('--eta-override must be positive')
        args.eta = args.eta_override

    args.eval_fine_to_coarse = None
    args.eval_coarse_names = None
    if args.eval_hierarchy_mapping is not None:
        with open(args.eval_hierarchy_mapping, encoding='utf-8') as handle:
            hierarchy = json.load(handle)
        args.eval_fine_to_coarse = [
            int(hierarchy['fine_to_coarse'][str(index)]) for index in range(args.ncls)
        ]
        coarse_count = max(args.eval_fine_to_coarse) + 1
        args.eval_coarse_names = hierarchy.get(
            'coarse_names', [f'{index:02d}' for index in range(coarse_count)]
        )
        if len(args.eval_coarse_names) != coarse_count:
            raise ValueError(
                'hierarchy coarse_names length does not match fine_to_coarse mapping'
            )
        if args.primary_eval_collapsed_coarse and args.ncls != 100:
            raise ValueError('collapsed coarse primary evaluation expects a 100-way student')
    
    # set up the train_dir and output_dir
    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.exp_name)
    print(args)
    return args

def is_special_epoch(epoch, total_epochs):
    in_last_80_percent = epoch >= int(total_epochs * 0.8)
    ends_with_9_or_last = (epoch % 10 == 9) or (epoch == total_epochs - 1)
    return in_last_80_percent and ends_with_9_or_last

def main():
    args = get_args()

    if args.train_seed is not None:
        random.seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"=> training seed: {args.train_seed}")
    
    # set up wandb
    if not args.disable_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed; pass --disable-wandb for an offline run")
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, entity="CVDD", dir="./")
        wandb.run.name = args.exp_name

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    print(args.original_data_path)
    assert os.path.exists(args.original_data_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
    if args.hard_label:
        train_dataset = datasets.ImageFolder(
            root=args.original_data_path,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(
                    size=args.input_size, scale=(args.min_scale, 1),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )
        if len(train_dataset.classes) != args.ncls:
            raise ValueError(
                f'hard-label ImageFolder has {len(train_dataset.classes)} classes, '
                f'expected {args.ncls}: {args.original_data_path}'
            )
        print(
            f'=> hard-label training: images={len(train_dataset)}, '
            f'classes={len(train_dataset.classes)}, loss=cross_entropy'
        )
    else:
        train_dataset = ImageFolder_FKD_MIX(
            fkd_path=args.fkd_path,
            mode=args.mode,
            args_epoch=args.epochs,
            args_bs=args.batch_size,
            root=args.original_data_path,
            transform=ComposeWithCoords(transforms=[
                RandomResizedCropWithCoords(size=args.input_size,
                                            scale=(args.min_scale, 1),
                                            interpolation=InterpolationMode.BILINEAR),
                RandomHorizontalFlipWithRes(),
                transforms.ToTensor(),
                normalize,
            ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)


    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.workers > 0,
    )
    if args.workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    train_loader = torch.utils.data.DataLoader(**loader_kwargs)

    # load validation data
    val_loader = load_val_loader(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))

    if args.model == 'ResNet18':
        if args.input_size <= 64:
            model = ResNet18(args.ncls)
        else:
            model = models.resnet18(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    elif args.model == 'ResNet50':
        if args.input_size <= 64:
            model = ResNet50(args.ncls)
        else:
            model = models.resnet50(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    elif args.model == 'ResNet101':
        if args.input_size <= 64:
            model = ResNet101(args.ncls)
        else:
            model = models.resnet101(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    else:
        raise ValueError('model not supported')
    model = model.cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs / args.eta)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)

 
    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)
        if not args.simple:
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                top1 = validate(model, args, epoch)
            else:
                top1 = 0
        else:
            if is_special_epoch(epoch, args.epochs):
                top1 = validate(model, args, epoch)
            else:
                top1 = 0
                
        if not args.disable_wandb:
            wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)

    if args.per_class_output is not None:
        best_path = os.path.join(args.output_dir, 'model_best.pth.tar')
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        export_per_class_accuracy(model, args, checkpoint['best_acc1'])

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    loss_function_ce = nn.CrossEntropyLoss()

    model.train()
    t1 = time.time()
    if hasattr(args.train_loader.dataset, 'set_epoch'):
        args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        if args.hard_label:
            images, target = batch_data
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            assert args.batch_size % args.gradient_accumulation_steps == 0
            small_bs = args.batch_size // args.gradient_accumulation_steps
            accum_step = math.ceil(images.shape[0] / small_bs)

            for accum_id in range(accum_step):
                partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
                partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
                output = model(partial_images)
                prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))
                loss = loss_function_ce(output, partial_target)
                loss = loss / accum_step
                loss.backward()

                n = partial_images.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

            optimizer.step()
            continue

        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        soft_label = soft_label.cuda(non_blocking=True).float()  # convert to float32
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            output = F.log_softmax(output/args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


        optimizer.step()


    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,}
    wandb_metrics.update(metrics)


    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()


def collapse_to_coarse(output, target, args):
    mapping = torch.tensor(args.eval_fine_to_coarse, dtype=torch.long, device=output.device)
    probabilities = torch.softmax(output, dim=1)
    coarse_probabilities = torch.zeros(
        output.shape[0], len(args.eval_coarse_names),
        dtype=probabilities.dtype, device=output.device,
    )
    coarse_probabilities.scatter_add_(
        1, mapping.unsqueeze(0).expand(output.shape[0], -1), probabilities
    )
    return coarse_probabilities, mapping[target]


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            output = model(data)
            if args.primary_eval_collapsed_coarse:
                evaluated_output, evaluated_target = collapse_to_coarse(output, target, args)
                loss = F.nll_loss(evaluated_output.clamp_min(1e-12).log(), evaluated_target)
            else:
                evaluated_output, evaluated_target = output, target
                loss = loss_function(output, target)

            prec1, prec5 = accuracy(evaluated_output, evaluated_target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def export_per_class_accuracy(model, args, best_acc1):
    model.eval()
    output_classes = len(args.eval_coarse_names) if args.primary_eval_collapsed_coarse else args.ncls
    correct = torch.zeros(output_classes, dtype=torch.long)
    total = torch.zeros(output_classes, dtype=torch.long)
    native_correct = 0
    native_total = 0
    with torch.no_grad():
        for data, target in args.val_loader:
            data = data.cuda(non_blocking=True)
            output = model(data)
            native_correct += output.argmax(1).cpu().eq(target.long()).sum().item()
            native_total += target.numel()
            if args.primary_eval_collapsed_coarse:
                evaluated_output, evaluated_target = collapse_to_coarse(
                    output, target.cuda(non_blocking=True), args
                )
                prediction = evaluated_output.argmax(1).cpu()
                target = evaluated_target.cpu().long()
            else:
                prediction = output.argmax(1).cpu()
                target = target.long()
            total.scatter_add_(0, target, torch.ones_like(target, dtype=torch.long))
            matched = prediction.eq(target).long()
            correct.scatter_add_(0, target, matched)
    classes = (args.eval_coarse_names if args.primary_eval_collapsed_coarse else
               getattr(args.val_loader.dataset, 'classes',
                       [str(index) for index in range(output_classes)]))
    per_class = []
    for class_id in range(output_classes):
        accuracy_value = 100.0 * correct[class_id].item() / max(total[class_id].item(), 1)
        per_class.append({
            'class_id': class_id,
            'class_name': classes[class_id],
            'correct': correct[class_id].item(),
            'total': total[class_id].item(),
            'accuracy': accuracy_value,
        })
    payload = {
        'best_top1': float(best_acc1),
        'training_target': ('hard_coarse_label' if args.hard_label else 'fkd_soft_label'),
        'num_classes': output_classes,
        'validation_dir': os.path.abspath(args.val_dir),
        'validation_images': len(args.val_loader.dataset),
        'primary_metric': ('collapsed_coarse20_top1'
                           if args.primary_eval_collapsed_coarse else 'native_top1'),
        'native_top1_at_best_checkpoint': 100.0 * native_correct / max(native_total, 1),
        'per_class': per_class,
    }
    output = os.path.abspath(args.per_class_output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f'Per-class best-checkpoint accuracy saved to: {output}', flush=True)

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
    if wandb is not None and wandb.run is not None:
        wandb.finish()
