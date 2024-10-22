"""
Training utilities for deep learning models with distributed training support.
This module provides functions for configuring training parameters, optimizers,
and managing training states across multiple devices.
"""

import torch
import os
import json
from dataclasses import dataclass
import random
import math
import numpy as np
from accelerate import Accelerator
from transformers import AdamW, get_scheduler
from typing import Dict, Optional


def set_deepspeed_config(accelerator: Accelerator, training_args: dataclass) -> None:
    """
    Configure DeepSpeed parameters for distributed training.

    Args:
        accelerator: Accelerator instance for distributed training
        training_args: Dataclass containing training configuration
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = training_args.per_device_train_batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        training_args.per_device_train_batch_size
        * world_size
        * accelerator.gradient_accumulation_steps
    )


def set_training_states(data_module: dict, training_args: dataclass) -> None:
    """
    Initialize all training states and parameters based on data and configuration.

    Args:
        data_module: Dictionary containing train and validation datasets
        training_args: Dataclass containing training configuration
    """
    set_num_steps_per_epoch(data_module, training_args)
    set_num_training_steps(training_args)
    set_num_updating_steps(training_args)
    set_num_eval_steps(training_args)
    set_per_eval_steps(training_args)
    set_num_warmup_steps(training_args)
    set_num_logging_steps(training_args)
    set_per_save_steps(training_args)

    print(
        f"+ [Training States] There are {training_args.num_training_steps} steps in total."
    )


def set_num_steps_per_epoch(data_module: dict, training_args: dataclass) -> None:
    """
    Calculate the number of steps per epoch for training and evaluation.
    Handles distributed training by considering the number of devices.

    Args:
        data_module: Dictionary containing train and validation datasets
        training_args: Dataclass containing training configuration
    """
    num_devices = int(os.environ.get("WORLD_SIZE", 1))

    len_train_set_per_device = math.ceil(
        len(data_module["train_dataset"]) / num_devices
    )
    num_train_steps_per_device = math.ceil(
        len_train_set_per_device / training_args.per_device_train_batch_size
    )
    num_updating_steps_per_epoch = (
        num_train_steps_per_device // training_args.gradient_accumulation_steps
    )

    len_eval_set_per_device = (
        math.ceil(len(data_module["val_dataset"]) / num_devices)
        if data_module["val_dataset"] is not None
        else None
    )
    num_eval_steps_per_device = (
        math.ceil(len_eval_set_per_device / training_args.per_device_eval_batch_size)
        if data_module["val_dataset"] is not None
        else None
    )

    training_args.num_training_steps_per_epoch = num_train_steps_per_device
    training_args.num_updating_steps_per_epoch = num_updating_steps_per_epoch
    training_args.num_eval_steps_per_epoch = num_eval_steps_per_device


def set_num_training_steps(training_args: dataclass) -> None:
    """Calculate total number of training steps across all devices."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if training_args.max_steps != -1:
        num_training_steps = training_args.max_steps
    else:
        assert training_args.num_train_epoches != -1
        num_training_steps = (
            training_args.num_training_steps_per_epoch * training_args.num_train_epoches
        )
    num_training_steps_aggr_devices = num_training_steps * world_size

    training_args.num_training_steps = num_training_steps
    training_args.num_training_steps_aggr_devices = num_training_steps_aggr_devices


def set_num_updating_steps(training_args: dataclass) -> None:
    """Calculate number of parameter update steps considering gradient accumulation."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    num_updating_steps = (
        training_args.num_training_steps // training_args.gradient_accumulation_steps
    )
    num_updating_steps_aggr_devices = num_updating_steps * world_size

    training_args.num_updating_steps = num_updating_steps
    training_args.num_updating_steps_aggr_devices = num_updating_steps_aggr_devices


def set_num_eval_steps(training_args: dataclass) -> None:
    """Set number of evaluation steps per epoch."""
    training_args.num_eval_steps = training_args.num_eval_steps_per_epoch


def set_per_eval_steps(training_args: dataclass) -> None:
    """Calculate frequency of evaluation during training."""
    if training_args.eval_steps != -1:
        per_eval_steps = training_args.eval_steps
    else:
        assert training_args.eval_epoches != -1
        per_eval_steps = (
            training_args.num_training_steps_per_epoch * training_args.eval_epoches
        )
    training_args.per_eval_steps = per_eval_steps


def set_num_warmup_steps(training_args: dataclass) -> None:
    """Calculate number of warmup steps for learning rate scheduler."""
    if training_args.warmup_steps != -1:
        num_updating_warmup_steps = training_args.warmup_steps
    else:
        assert training_args.warmup_ratio != -1
        num_updating_warmup_steps = int(
            training_args.num_updating_steps * training_args.warmup_ratio
        )
    num_updating_warmup_steps_aggr_devices = num_updating_warmup_steps * int(
        os.environ.get("WORLD_SIZE", 1)
    )

    training_args.num_updating_warmup_steps = num_updating_warmup_steps
    training_args.num_updating_warmup_steps_aggr_devices = (
        num_updating_warmup_steps_aggr_devices
    )


def set_num_logging_steps(training_args: dataclass) -> None:
    """Calculate frequency of logging during training."""
    if training_args.logging_steps != -1:
        num_logging_steps = training_args.logging_steps
    else:
        assert training_args.logging_epoches != -1
        num_logging_steps = (
            training_args.num_training_steps_per_epoch * training_args.logging_epoches
        )
    training_args.num_logging_steps = num_logging_steps


def set_per_save_steps(training_args: dataclass) -> None:
    """Calculate frequency of model checkpointing during training."""
    if training_args.save_steps != -1:
        per_save_steps = training_args.save_steps
    else:
        assert training_args.save_epoches != -1
        per_save_steps = (
            training_args.num_training_steps_per_epoch * training_args.save_epoches
        )
    training_args.per_save_steps = per_save_steps


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_optimizers(
    model: "transformers.AutoModelForCausalLM", training_args: dataclass
) -> Dict:
    """
    Create optimizer and learning rate scheduler for model training.

    Args:
        model: The transformer model to optimize
        training_args: Dataclass containing training configuration

    Returns:
        Dict containing optimizer and learning rate scheduler
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
    )

    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_updating_warmup_steps_aggr_devices,
        num_training_steps=training_args.num_updating_steps_aggr_devices,
    )
    return optimizer, lr_scheduler
