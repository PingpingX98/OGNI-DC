from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.ognidc import OGNIDC

from summary.gcsummary import OGNIDCSummary
from summary.gcsummarynew import OGNIDCSummarynew
from metric.dcmetric import DCMetric
from metric.dcmetricnew import DCMetricnew
from data import get as get_data
from loss.sequentialloss import SequentialLoss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import torch.utils.benchmark as benchmark
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.cuda.amp as amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            # new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1

    return new_args



def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    if args.model == 'OGNIDC':
        loss = SequentialLoss(args)
        #summ = OGNIDCSummary
        summ_new = OGNIDCSummarynew
    else:
        raise NotImplementedError

    # Network
    if args.model == 'OGNIDC':
        net = OGNIDC(args)
    else:
        raise TypeError(args.model, ['OGNIDC', ])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    #metric = DCMetric(args)
    metric_new = DCMetricnew(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    #writer_test = summ(args.save_dir, 'test', args, None, metric.metric_name)
    writer_test_new = summ_new(args.save_dir, 'test', args, None, metric_new.metric_name)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0
    times = []
    init_seed()
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}
        torch.cuda.reset_max_memory_allocated(device=0)
        timer = benchmark.Timer(
            stmt='net(sample)',
            globals={'net': net, 'sample': sample}
        )
        
        measured_time = timer.timeit(1000).mean
        break
    peak_memory = torch.cuda.max_memory_allocated(device=0) / 1024 ** 3
    print(f"[Batch {batch}] Inference time: {measured_time:.6f}s | Peak memory: {peak_memory:.3f} GB")
    model = net.module if isinstance(net, torch.nn.DataParallel) else net
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
        

def main(args):
    init_seed()
    if not args.test_only:
        if not args.multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        # args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir, args.epochs)
        args.pretrain = '{}/model_best.pt'.format(args.save_dir)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)