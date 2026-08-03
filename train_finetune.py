"""Supervised fine-tuning for multi-label biomarker classification.

Initializes the image aggregator from MARBLE pretrained weights (or randomly
for the supervised baseline) and trains a classification head.

Usage:
    torchrun --nproc_per_node=4 train_finetune.py --config configs/finetune.yaml
"""

import argparse
import datetime
import logging
import os
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from marble.aggregator import Agata, FCHeadConfig, LinearLayerSpec
from marble.dataset import FinetuneDataset
from marble.loss import masked_bce_loss
from marble.model import FinetuneModel
from marble.utils import (
    compute_auc_per_class,
    load_pretrained_weights,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def setup_ddp():
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=1800))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.barrier()
    return local_rank


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    tensor = tensor.to(torch.cuda.current_device())
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    set_seed(42)

    local_rank = setup_ddp()

    # Load data — adapt this to your data loading pipeline
    from marble.utils import load_data_panel_version

    train_df, biom_panel, _ = load_data_panel_version(config, mode="train")
    tune_df, _, _ = load_data_panel_version(config, mode="tune")

    num_classes = len(biom_panel)

    train_dataset = FinetuneDataset(train_df, biom_panel, config.dataset.path_to_bags, config.dataset.max_len)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, sampler=train_sampler,
        pin_memory=True, drop_last=True,
    )

    tune_dataset = FinetuneDataset(tune_df, biom_panel, config.dataset.path_to_bags.replace("train", "tune"), config.dataset.max_len)
    tune_sampler = DistributedSampler(tune_dataset, shuffle=False)
    tune_loader = DataLoader(
        tune_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, sampler=tune_sampler,
        pin_memory=True, drop_last=True,
    )

    # Model
    label_name_fclayer_head_config: Mapping[str, FCHeadConfig] = {
        "biomarkers": FCHeadConfig(
            in_channels=config.model.layer2_out,
            layer_specs=[LinearLayerSpec(dim=num_classes, activation=nn.Sigmoid())],
        )
    }

    image_agg = Agata(
        in_features=config.model.in_features,
        num_classes=num_classes,
        layer1_out_features=config.model.layer1_out,
        layer2_out_features=config.model.layer2_out,
        activation=nn.ReLU(),
        label_name_fclayer_head_config=label_name_fclayer_head_config,
        n_attention_queries=config.model.n_attention_queries,
    ).cuda()

    model = FinetuneModel(
        image_encoder=image_agg, image_embedding=config.model.layer2_out, num_classes=num_classes,
    )

    # Load pretrained weights if available
    if hasattr(config.model, "pretrained_checkpoint") and config.model.pretrained_checkpoint:
        model = load_pretrained_weights(model, config.model.pretrained_checkpoint, strict=False)
        if dist.get_rank() == 0:
            print(f"Loaded pretrained weights from {config.model.pretrained_checkpoint}")

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)

    os.makedirs(config.model.checkpoint_dir, exist_ok=True)

    best_auc = 0.0
    for epoch in range(config.training.num_epochs):
        # --- Train ---
        model.train()
        train_sampler.set_epoch(epoch)
        losses = []

        for batch_idx, (bags, masks, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}", disable=(dist.get_rank() != 0))):
            logits = model(bags.cuda(), masks.cuda())
            loss = masked_bce_loss(logits, labels.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bags, masks, labels in tqdm(tune_loader, desc=f"Val Epoch {epoch}", disable=(dist.get_rank() != 0)):
                logits = model(bags.cuda(), masks.cuda())
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Gather across ranks
        preds_tensor = gather_tensor(torch.tensor(all_preds, dtype=torch.float32)).cpu().numpy()
        labels_tensor = gather_tensor(torch.tensor(all_labels, dtype=torch.float32)).cpu().numpy()

        scheduler.step()

        if dist.get_rank() == 0:
            auc_scores = compute_auc_per_class(labels_tensor, preds_tensor)
            mean_auc = np.nanmean(auc_scores)
            print(f"Epoch {epoch} | Train loss: {np.mean(losses):.4f} | Val AUC: {mean_auc:.4f}")

            if mean_auc > best_auc:
                best_auc = mean_auc
                save_checkpoint(
                    model, optimizer, epoch, np.mean(losses),
                    path=f"{config.model.checkpoint_dir}/best_checkpoint_auc_{best_auc:.4f}.pth",
                )
                print(f"New best AUC: {best_auc:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
