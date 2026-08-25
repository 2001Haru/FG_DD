import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX,
                       RandomHorizontalFlipWithRes,
                       RandomResizedCropWithCoords, mix_aug, load_model,count_jpg_files)
import platform
import sys
# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import recover.utils_recover as ure

parser = argparse.ArgumentParser(description='FKD Soft Label Generation w/ Mix Augmentation')
parser.add_argument('--syn-data-path', required=True, type=str,
                    help='the path to the syn data which is being processed in this relabeling process')
parser.add_argument('--online', action='store_true',
                    help='use online model')
parser.add_argument('--multi-model', action='store_true',
                    help='use multi teacher model')
parser.add_argument('--model-choice', nargs='+', 
                    help='A list containing the choices of the compare model')
parser.add_argument('--model-weight', nargs='+', 
                    help='A list containing the choices of the compare model')
parser.add_argument('--eval-mode', type=str,default="F",
                    help='whether to use the evaluation mode or not')
parser.add_argument('--teacher-model-name', type=str,
                    help='teacher model name')
parser.add_argument('--teacher-num-classes', type=int, default=None,
                    help='Teacher output dimension when it differs from dataset-name')
parser.add_argument('--teacher-mapping', type=str, default=None,
                    help='fine_to_coarse hierarchy for temperature-compatible probability marginalization')
parser.add_argument('--marginalize-temperature', type=float, default=20.0,
                    help='temperature at which Teacher probabilities are marginalized')
parser.add_argument('--model-pool-dir', type=str, default=None,
                    help='required when pretrained model type is offline, the directory of the models when using offline mode')
parser.add_argument('--fkd-path',required=True, type=str,
                    help='the path to save the fkd soft labels')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--persistent-workers', action='store_true',
                    help='keep DataLoader workers alive across relabel epochs')
parser.add_argument('--prefetch-factor', type=int, default=2)
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--dataset-name', default='cifar100', type=str,
                    help='dataset name')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.62.156:23457', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FKD soft label generation args
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument("--min-scale-crops", type=float, default=0.08,
                    help="argument in RandomResizedCrop")
parser.add_argument("--max-scale-crops", type=float, default=1.,
                    help="argument in RandomResizedCrop")
parser.add_argument('--use-fp16', dest='use_fp16', action='store_true',
                    help='save soft labels as `fp16`')
parser.add_argument('--mode', default='fkd_save', type=str, metavar='N',)
parser.add_argument('--fkd-seed', default=42, type=int, metavar='N')
parser.add_argument('--mix-type', default = None, type=str, choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')

def set_worker_sharing_strategy(worker_id: int) -> None:
    if platform.system() == 'Linux':
        sharing_strategy = 'file_descriptor'
    else:
        sharing_strategy = 'file_system'
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    args = parser.parse_args()
    
    # set up the mean, std and ncls for the dataset
    if args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.input_size = 32
    elif args.dataset_name == 'cifar10':
        args.mean_norm = [0.4914, 0.4822, 0.4465]
        args.std_norm = [0.2470, 0.2435, 0.2616]
        args.ncls = 10
        args.input_size = 32
    elif args.dataset_name == 'cifar20':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 20
        args.input_size = 32
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.input_size = 224
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.input_size = 224
    elif args.dataset_name == 'imagewoof':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.input_size = 224
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.jitter = 4
        args.input_size = 64
    elif args.dataset_name == 'imagenet100':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 100
        args.jitter = 32
        args.input_size = 224
    elif args.dataset_name == 'CUB_imsize224':
        args.mean_norm = [0.4857, 0.4994, 0.4326]
        args.std_norm = [0.2260, 0.2215, 0.2595]
        args.ncls = 200
        args.jitter = 32
        args.input_size = 224
    else:
        raise ValueError('dataset not supported')
    
    
    # compute current ipc
    image_count = count_jpg_files(args.syn_data_path)
    if image_count <= 0 or image_count % args.ncls != 0:
        raise RuntimeError(
            f'invalid relabel ImageFolder image count: images={image_count}, '
            f'classes={args.ncls}, path={args.syn_data_path}'
        )
    ipc = image_count // args.ncls
    
    # set up the fkd path
    args.fkd_path = args.fkd_path + f'_bs{args.batch_size}_ipc{ipc}'
    if not os.path.exists(args.fkd_path):
        os.makedirs(args.fkd_path, exist_ok=True)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.gpu)
    # load different teacher models
    args.teacher_to_target = None
    if args.teacher_mapping is not None:
        with open(args.teacher_mapping, encoding='utf-8') as handle:
            hierarchy = json.load(handle)
        teacher_classes = args.teacher_num_classes or args.ncls
        args.teacher_to_target = torch.tensor([
            int(hierarchy['fine_to_coarse'][str(index)])
            for index in range(teacher_classes)
        ], dtype=torch.long)
        if max(args.teacher_to_target.tolist()) + 1 != args.ncls:
            raise RuntimeError('teacher mapping target count does not match dataset ncls')
    teacher_model_lis = []
    if args.multi_model:
        for model_name in args.model_choice:
            if args.online:
                model = ure.load_online_model(model_name, args)
            else:
                model = load_model(args, model_name)
            teacher_model_lis.append(model)
    else:
        print(f"Teacher model name: {args.teacher_model_name}")
        if args.online:
            model = ure.load_online_model(args.teacher_model_name, args)
        else:
            model = load_model(args,args.teacher_model_name)
        teacher_model_lis.append(model)
    
    print(f"Total model amount: {len(teacher_model_lis)}")
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            for _model in teacher_model_lis:
                _model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                _model = torch.nn.parallel.DistributedDataParallel(_model, device_ids=[args.gpu])
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            for _model in teacher_model_lis:
                _model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                _model = torch.nn.parallel.DistributedDataParallel(_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for _model in teacher_model_lis:
            _model = _model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        for _model in teacher_model_lis:
            _model = torch.nn.DataParallel(_model).cuda()

    # freeze all layers
    for _model in teacher_model_lis:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    cudnn.benchmark = args.seed is None

    print("process data from {}".format(args.syn_data_path))

    normalize = transforms.Normalize(mean=args.mean_norm,
                                     std=args.std_norm)
    
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops,
                                               args.max_scale_crops),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    loader_options = dict(
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
        persistent_workers=args.persistent_workers and args.workers > 0,
    )
    if args.workers > 0:
        loader_options['prefetch_factor'] = args.prefetch_factor
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        **loader_options)
    
    if args.eval_mode == 'T':
        for model in teacher_model_lis:
            model.eval()
        print('Not Applying BSSL')
    else:
        # BSSL requires batch statistics from the current synthetic batch.
        # Make the intended state explicit instead of relying on nn.Module's
        # default construction state.
        for model in teacher_model_lis:
            model.train()
        if not all(model.training for model in teacher_model_lis):
            raise RuntimeError('BSSL requires every Teacher to be in train mode')
        print("Applying BSSL: Teacher train mode=True")

    for epoch in tqdm([i for i in range(args.epochs)]):
        dir_path = os.path.join(args.fkd_path, 'epoch_{}'.format(epoch))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save(train_loader, teacher_model_lis, dir_path, args)
        # exit()


@torch.no_grad()
def save(train_loader, model_lis, dir_path, args):
    if args.model_weight is None:
        weights = [1.0 / len(model_lis)] * len(model_lis)
    else:
        w = np.array([float(w) for w in args.model_weight])
        temperature = 10
        w = w / temperature
        weights = np.exp(w) / np.sum(np.exp(w))
        

    """Generate soft labels and save"""
    profile_started = time.time()
    compute_seconds = 0.0
    save_seconds = 0.0
    batches = 0
    for batch_idx, (images, target, flip_status, coords_status) in enumerate(train_loader):
        compute_started = time.time()
        images = images.cuda(non_blocking=True)
        split_point = int(images.shape[0] // 2)
        origin_images = images
        images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)
        
        total_output = []
        for idx, _model in enumerate(model_lis):
            cat_output = []
            output = _model(origin_images[:split_point])
            cat_output.append(output)
            output = _model(origin_images[split_point:])
            cat_output.append(output)
            output = torch.cat(cat_output, 0) * weights[idx]
            total_output.append(output)
            
        output = torch.stack(total_output, 0)
        output = output.sum(0)
        if args.teacher_to_target is not None:
            temperature = args.marginalize_temperature
            probabilities = torch.softmax(output / temperature, dim=1)
            target_probabilities = torch.zeros(
                output.shape[0], args.ncls, dtype=probabilities.dtype,
                device=probabilities.device,
            )
            mapping = args.teacher_to_target.to(probabilities.device)
            target_probabilities.scatter_add_(
                1, mapping.unsqueeze(0).expand(output.shape[0], -1), probabilities
            )
            output = temperature * target_probabilities.clamp_min(1e-12).log()
        
        if args.use_fp16:
            output = output.half()
        output_cpu = output.cpu()
        compute_seconds += time.time() - compute_started

        batch_config = [coords_status, flip_status, mix_index, mix_lam, mix_bbox, output_cpu]
        batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
        save_started = time.time()
        torch.save(batch_config, batch_config_path)
        save_seconds += time.time() - save_started
        batches += 1
    total_seconds = time.time() - profile_started
    data_wait_seconds = max(0.0, total_seconds - compute_seconds - save_seconds)
    print(
        f"RELABEL_PROFILE batches={batches} total={total_seconds:.3f}s "
        f"data_wait={data_wait_seconds:.3f}s teacher_compute={compute_seconds:.3f}s "
        f"save_io={save_seconds:.3f}s",
        flush=True,
    )


if __name__ == '__main__':
    main()
