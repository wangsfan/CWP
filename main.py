# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import logging

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets_n import build_continual_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')



def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask, class_list = build_continual_dataloader(args)
    print(class_list)
    print("NB CLasses: ", args.nb_classes)

    print(f"Creating old model: {args.model}")
    model_old = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args,
        class_list=class_list
    )

    model_old.to(device)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        num_tasks=args.num_tasks,
        kernel_size=args.kernel_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        prompts_per_task=args.num_prompts_per_task,
        args=args,
        class_list=class_list
    )
    model.to(device)  

    if args.freeze:
        for p in model_old.parameters():
            p.requires_grad = False
        
        for n, p in model.named_parameters():
            if n.startswith('key_task'):
                if n.startswith('key_task.ct_class'):
                    p.requires_grad = True
                if n.startswith('key_task.ctx'):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            if n.startswith('text_encoder'):     
                p.requires_grad = False
            if n.startswith('class_name'):     
                p.requires_grad = False
            if n.startswith('task_name'):     
                p.requires_grad = False  
                
            if n.startswith(tuple(args.freeze)):
                if n.find('norm1')>=0 or n.find('norm2')>=0:
                    pass
                else:
                    p.requires_grad = False

        # exit(0)
        
    
    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            if task_id>0:
                model.head.update(len(class_mask[task_id]))
            checkpoint_path = os.path.join(args.output_dir+ '_'+ args.dataset, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, model_old, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    criterion = torch.nn.CrossEntropyLoss().to(device)

    milestones = [18] if "CIFAR" in args.dataset else [40]
    lrate_decay = 0.1
    param_list = list(model.parameters())
 

    network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    
    if not args.SLCA:
        optimizer = create_optimizer(args, model)
        if args.sched != 'constant':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.sched == 'constant':
            lr_scheduler = None
    else:
        optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model,model_old,
                    criterion, data_loader, lr_scheduler, optimizer,
                    device, class_mask, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    print("Started main")
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    print("Parser created: ", parser)
    
    print("Getting config")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_cwprompt':
        from configs.cifar100_cwprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_cwprompt', help='Split-CIFAR100 configs for cwprompt')
    elif config == 'imr_cwprompt':
        from configs.imr_cwprompt import get_args_parser
        config_parser = subparser.add_parser('imr_cwprompt', help='Split-ImageNet-R configs for cwprompt')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R configs for DualPrompt')
    elif config == 'cub_cwprompt':
        from configs.cub_cwprompt import get_args_parser
        config_parser = subparser.add_parser('cub_cwprompt', help='Split-CUB configs for cwprompt')
    elif config == 'cifar100_slca':
        from configs.cifar100_slca import get_args_parser
        config_parser = subparser.add_parser('cifar100_slca', help='Split-CIFAR100 SLCA configs')
    elif config == 'imr_slca':
        from configs.imr_slca import get_args_parser
        config_parser = subparser.add_parser('imr_slca', help='Split-ImageNet-R SLCA configs')
    elif config == 'cub_slca':
        from configs.cub_slca import get_args_parser
        config_parser = subparser.add_parser('cub_slca', help='Split-CUB SLCA configs')
    elif config == 'eurosat_cwprompt':
        from configs.eurosat_cwprompt import get_args_parser
        config_parser = subparser.add_parser('eurosat_cwprompt', help='Split-EuroSAT configs')    
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    print("Reached here")
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Reached here")
    main(args)
    
    sys.exit(0)
