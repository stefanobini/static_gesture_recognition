import math
import sys
import time
import torch
import utils
import cv2
import torchvision.transforms as T
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.profiler import profile, record_function, ProfilerActivity
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.7f}"))
    header = f"Epoch: [{epoch}]"

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
            

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        lr_value = optimizer.param_groups[0]["lr"]

    return losses, loss_dict, lr_value


@torch.inference_mode()
def evaluate(model, data_loader, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"


    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            loss_dict = model(images, targets)
            
            losses_reduced = sum(loss for loss in loss_dict.values())
        
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)
        metric_logger.update(loss=losses_reduced, **loss_dict)

    return loss_dict


def benchmark (model, data_loader, device,):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Benchmark: "
    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                model(images)

        
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

        print(count_parameters(model))
        print("")
        print("cuda_time_total")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("cuda_memory_usage")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        break

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_total(model):
    return sum(p.numel() for p in model.parameters())