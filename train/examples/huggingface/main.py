#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Source: `pytorch imagenet example <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_ # noqa B950

Modified and simplified to make the original pytorch example compatible with
torchelastic.distributed.launch.

Changes:

1. Removed ``rank``, ``gpu``, ``multiprocessing-distributed``, ``dist_url`` options.
   These are obsolete parameters when using ``torchelastic.distributed.launch``.

2. Removed ``seed``, ``evaluate``, ``pretrained`` options for simplicity.

3. Removed ``resume``, ``start-epoch`` options.
   Loads the most recent checkpoint by default.

4. ``batch-size`` is now per GPU (worker) batch size rather than for all GPUs.

5. Defaults ``workers`` (num data loader workers) to ``0``.

Usage

::

 >>> python -m torchelastic.distributed.launch
        --nnodes=$NUM_NODES
        --nproc_per_node=$WORKERS_PER_NODE
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        main.py
        --arch resnet18
        --epochs 20
        --batch-size 32
        <DATA_DIR>
"""

import argparse
import io
import os
import shutil
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import List, Tuple

import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import SGD

from datasets import load_dataset, Features, ClassLabel, Value, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AdamW

from pathlib import Path
import pandas as pd

from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from collections import OrderedDict
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

import yaml, wandb
from ruamel.yaml import YAML

def load_yaml_config():
    "Load config for training (params not touched by the sweep)"
    yaml = YAML()
    yaml.preserve_quotes = True    
    # Read train template yaml
    with open('./train_template.yaml') as file:
        train_template = yaml.load(file)



def run(args):
    wandb_run = wandb.init(config=args)
    args = wandb.config

    device_id = int(os.environ["LOCAL_RANK"])
        
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")

    dist.init_process_group(
        backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=120)
    )

    model, criterion, optimizer = initialize_huggingface_model(
        args.arch, args.lr, args.momentum, args.weight_decay, args.optimizer, device_id
    )

    train_loader, test_loader = initialize_custom_data_loader(
        args.data, args.batch_size, args.workers
    )
    
    # resume from checkpoint if one exists;
    state = load_checkpoint(
        args.checkpoint_file, device_id, args.arch, model, optimizer
    )

    start_epoch = state.epoch + 1
    print(f"=> start_epoch: {start_epoch}, best_acc1: {state.best_acc1}")

    #convergence_df = pd.DataFrame(columns = ['Epochs','Train_Loss','Validation_Loss'])

    training_start_time = time.time()

    print_freq = args.print_freq
    for epoch in range(start_epoch, args.epochs):
        state.epoch = epoch
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        epoch_start_time = time.time()
        print('Starting Training Epoch')
        train_loss_epoch = train(train_loader, model, criterion, optimizer, epoch, device_id, print_freq)
        print('Epoch finished, took {:.2f}s'.format(time.time() - epoch_start_time))

        # evaluate on validation set
        # val_start_time = time.time()
        # print('Starting Validation')
        # val_loss_epoch = validate(val_loader, model, criterion, device_id, print_freq)
        # print('Validation finished, took {:.2f}s'.format(time.time() - val_start_time))

        train_loss = sum(train_loss_epoch)/len(train_loss_epoch)
        # val_loss = sum(val_loss_epoch)/len(val_loss_epoch)
        val_loss = 100

        # remember best loss@1 and save checkpoint
        is_best = val_loss < state.best_acc1
        state.best_acc1 = min(val_loss, state.best_acc1)

        # convergence_df.loc[epoch,'Epochs'] = epoch
        # convergence_df.loc[epoch,'Train_Loss'] = train_loss
        # convergence_df.loc[epoch,'Validation_Loss'] = val_loss


        if device_id == 0:
            save_checkpoint(state, is_best, args.checkpoint_file)
            # convergence_df.to_csv('/shared-efs/arxiv/convergence_df.csv')

    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
    
    run_predictions(args.checkpoint_file, args.lr,args.optimizer)

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Elastic HuggingFace Training")
    
    # Required paramaters
    parser.add_argument(
        "data", 
        metavar="DIR", 
        help="path to dataset",
        required=True
    )
    parser.add_argument(
        "--sweep_id", 
        default=None,
        help="The Sweep id created by wandb",
    )

    # Other params
    parser.add_argument("--arch", default="HuggingFace")
    parser.add_argument("--workers", default=0, help="number of data loading workers")
    parser.add_argument("--epochs", default=10, help="number of total epochs to run")
    parser.add_argument("--batch-size", default=32, help="mini-batch size per worker (GPU)")
    parser.add_argument("--lr", default=5e-5, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--print-freq", default=1, help="print frequency (default: 10)")
    parser.add_argument("--dist-backend", default="nccl", choices=["nccl", "gloo"], help="distributed backend")
    parser.add_argument("--checkpoint-file", default="/shared-efs/checkpoint.pth.tar", help="checkpoint file path, to load and save to")
    parser.add_argument("--optimizer", default="AdamW", help="optimizer type")
    args = parser.parse_args()

    # config = load_yaml_config()

    if args.sweep_id is not None:
        wandb.agent(sweep_id, run(args), count=5)    
    
        

class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, file_name ):
    
        #'Initialization'
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir+file_name)
        
    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.df)
        
    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        df = self.df
        one_line = df['Text'][index]
        label = df['labels'][index]
        
        return (one_line,label)

def collate_tokenize(data,tokenizer):
    text_batch = [element[0] for element in data]
    labels = [element[1] for element in data]
    tokenized_inputs = tokenizer(text_batch, padding='max_length', truncation=True, return_tensors='pt')
    
    tokenized_inputs['labels'] = torch.tensor(labels)
    tokenized_inputs['attention_mask'] = tokenized_inputs['attention_mask']

    return tokenized_inputs
    
class MyCollator(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        # do something with batch and self.params
        tokenized_inputs = collate_tokenize(batch,self.tokenizer)
        
        return tokenized_inputs


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 10
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def initialize_huggingface_model(
    arch: str, lr: float, momentum: float, weight_decay: float, optimizer_type, device_id: int
):
    print(f"=> creating model: {arch}")
    
    ## Initializing the model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    
    model.cuda(device_id)
    
    cudnn.benchmark = True
    
    model = DistributedDataParallel(model, device_ids=[device_id])
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)

    # initialize optimizer
    if optimizer_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return model, criterion, optimizer


def initialize_custom_data_loader(
    data_dir, batch_size, num_data_workers
) -> Tuple[DataLoader, DataLoader]:
    
    # Generators
    train_dataset = Dataset(data_dir, file_name = 'train.csv')
    print('Train dataset done')

    train_sampler = ElasticDistributedSampler(train_dataset)
    print('Train sampler done')
    
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    my_collator = MyCollator(tokenizer)
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_data_workers,
            pin_memory=True,
            collate_fn=my_collator,
            sampler=train_sampler
        )
        
    print('Train loader done')
    
    test_dataset = Dataset(data_dir, file_name = 'test.csv')
    
    print('Test dataset done')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        collate_fn=my_collator
    )
    
    print('Test loader done')
    
    return train_loader, test_loader
    


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: DistributedDataParallel,
    optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    # with tmp_process_group(backend="gloo") as pg:
    #     rank = dist.get_rank(group=pg)

    #     # get rank that has the largest state.epoch
    #     epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
    #     epochs[rank] = state.epoch
    #     dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
    #     t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
    #     max_epoch = t_max_epoch.item()
    #     max_rank = t_max_rank.item()

    #     # max_epoch == -1 means no one has checkpointed return base state
    #     if max_epoch == -1:
    #         print(f"=> no workers have checkpoints, starting from epoch 0")
    #         return state

    #     # broadcast the state from max_rank (which has the most up-to-date state)
    #     # pickle the snapshot, convert it into a byte-blob tensor
    #     # then broadcast it, unpickle it and apply the snapshot
    #     print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

    #     with io.BytesIO() as f:
    #         torch.save(state.capture_snapshot(), f)
    #         raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

    #     blob_len = torch.tensor(len(raw_blob))
    #     dist.broadcast(blob_len, src=max_rank, group=pg)
    #     print(f"=> checkpoint broadcast size is: {blob_len}")

    #     if rank != max_rank:
    #         blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
    #     else:
    #         blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

    #     dist.broadcast(blob, src=max_rank, group=pg)
    #     print(f"=> done broadcasting checkpoint")

    #     if rank != max_rank:
    #         with io.BytesIO(blob.numpy()) as f:
    #             snapshot = torch.load(f)
    #         state.apply_snapshot(snapshot, device_id)

    #     # wait till everyone has loaded the checkpoint
    #     dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    #tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), filename)
    #os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)


def train(
    train_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    optimizer,  # AdamW,
    epoch: int,
    device_id: int,
    print_freq: int
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    #top1 = AverageMeter("Acc@1", ":6.2f")
    #top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    print('Length of train_loader = ' + str(len(train_loader)))

    # switch to train mode
    model.train()
    


    end = time.time()
    train_loss_epoch = []
    i = 0
    for (idx,batch) in enumerate(train_loader):
        #print(i)
        # measure data loading time
        data_time.update(time.time() - end)

        
        optimizer.zero_grad()
        input_ids = batch['input_ids'].cuda(device_id, non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(device_id, non_blocking=True)
        labels = batch['labels'].cuda(device_id, non_blocking=True)
        
        #print('Len of input_ids  = ' + str(len(input_ids)))
        
        
         # compute output
        outputs = model(input_ids, attention_mask=attention_mask,labels = labels)
        
        #print('Output done')
        
        loss = outputs[0]
        train_loss_epoch.append(loss.item())

        # # measure accuracy and record loss
        losses.update(loss.item(),input_ids.size(0))
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
        i = i + 1

    return train_loss_epoch

        


def validate(
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    #top1 = AverageMeter("Acc@1", ":6.2f")
    #top5 = AverageMeter("Acc@5", ":6.2f")
    # progress = ProgressMeter(
    #     len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    # )
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()

        val_loss_epoch = []
        i = 0
        for (idx,batch) in enumerate(val_loader):
        
            if device_id is not None:
                input_ids = batch['input_ids'].cuda(device_id, non_blocking=True)

            attention_mask = batch['attention_mask'].cuda(device_id, non_blocking=True)
            labels = batch['labels'].cuda(device_id, non_blocking=True)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels = labels)

            # compute output
            loss = outputs[0]
            val_loss_epoch.append(loss.item())

            # # measure accuracy and record loss
            losses.update(loss.item(),input_ids.size(0))
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            #     progress.display(i)
            i = i + 1

        # TODO: this should also be done with the ProgressMeter
        #print("Loss = %.3f" % loss.item())
        # print(
        #     " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        # )

    return val_loss_epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch: int, lr: float) -> None:
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


# def accuracy(output, target, topk=(1,)):
#     """
#     Computes the accuracy over the k top predictions for the specified values of k
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def run_predictions(
    checkpoint_filename,
    lr,
    optimizer
):
    
    checkpoint_dir = os.path.dirname(checkpoint_filename)
    print('*****************')
    print(checkpoint_dir)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_dir+"/checkpoint.tar", map_location=str(device))
    state_dict = checkpoint['state_dict']
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        #name = k[7:] # remove `module.`
        name = k.replace('module.', '') # removing ‘moldule.’ from key
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()
    
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_df = pd.read_csv('/shared-efs/wandb-finbert/test.csv')
    
    tokenized_test_inputs = tokenizer(list(test_df['Text']), padding='max_length', truncation=True, return_tensors='pt')
    
    # tokenized_test_inputs.to(device)
    # model.to(device)
    
    with torch.no_grad():
        preds = model(**tokenized_test_inputs)
        
    pred_labels = preds[0].argmax(axis = 1).tolist()
    true_labels = list(test_df['labels'])
    
    recall = recall_score(y_pred=pred_labels,y_true = true_labels)
    f1 = f1_score(y_pred=pred_labels,y_true = true_labels)
    accuracy = accuracy_score(y_pred=pred_labels,y_true = true_labels)
    precision = precision_score(y_pred=pred_labels,y_true = true_labels)
    
    pred_dict = {'recall': recall,
    'f1': f1,
    'accuracy': accuracy,
    'precision': precision}
    
    metrics_df = pd.DataFrame()
    metrics_df = metrics_df.append(pred_dict, ignore_index=True)
    
    run_name = checkpoint_dir.split('/')[-1]
    metrics_df['run_name'] = run_name
    metrics_df['lr'] = lr
    metrics_df['optimizer'] = optimizer
    
    
    if os.path.exists('/shared-efs/wandb-finbert/all_results.csv'):
        metrics_df.to_csv('/shared-efs/wandb-finbert/all_results.csv', mode='a', index=False, header=False)
    else:
        metrics_df.to_csv('/shared-efs/wandb-finbert/all_results.csv', index=False)
    
    return None


if __name__ == "__main__":
    main()