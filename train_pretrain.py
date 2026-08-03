"""MARBLE-LLM contrastive pretraining (Scenario 1: LLM-only alignment).

Usage:
    torchrun --nproc_per_node=4 train_pretrain.py --config configs/pretrain_llm.yaml
"""

import argparse
import datetime
import logging
import os
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from marble.aggregator import Agata, FCHeadConfig, LinearLayerSpec
from marble.dataset import PretrainDataset_LLM
from marble.loss import contrastive_loss_ddp
from marble.model import CrossAttentionAggregator, MARBLEModel_LLM
from marble.utils import load_data_panel_version, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MARBLE-LLM contrastive pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def setup_ddp():
    master_uri = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(seconds=1800),
        init_method=master_uri,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.barrier()
    return local_rank


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    set_seed(42)

    # Load data
    train_df, biom_panel, impact_version = load_data_panel_version(config, mode="train")
    tune_df, _, _ = load_data_panel_version(config, mode="tune")

    panel_label_mapping = pd.read_csv(config.dataset.panel_label_mapping)
    descriptions = panel_label_mapping["biomarker_description"]
    after_first_word = descriptions.str.split(" ", n=1).str[1]
    cgl = pd.read_csv(config.dataset.cancer_gene_list, sep="\t", header=0)
    gene_names = cgl["Hugo Symbol"].tolist()

    # Setup DDP
    local_rank = setup_ddp()

    # Load precomputed LLM embeddings
    with open(config.dataset.embeds_tumor_gene_path, "rb") as f:
        embeds_llm = pickle.load(f)

    # Datasets and dataloaders
    train_dataset = PretrainDataset_LLM(
        train_df, biom_panel, config.dataset.path_to_bags,
        embeds_llm, descriptions, after_first_word, gene_names, impact_version,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, sampler=train_sampler,
        pin_memory=True, drop_last=True,
    )

    # Model
    num_classes = config.model.num_classes
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

    biomarker_encoder = CrossAttentionAggregator(384).cuda()
    model = MARBLEModel_LLM(
        image_encoder=image_agg, biom_encoder=biomarker_encoder, text_embedding=384,
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)

    # Logging
    os.makedirs(config.model.checkpoint_dir, exist_ok=True)
    os.makedirs(config.model.log_dir, exist_ok=True)
    log_file = Path(config.model.log_dir) / "training_log.txt"
    logging.basicConfig(filename=log_file, filemode="a", format="%(asctime)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger()

    # Training loop
    for epoch in range(config.training.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.get_rank() != 0))
        for batch_idx, (bags, masks, labels, biom_embeds, biom_masks, metadata) in enumerate(pbar):
            bags, masks = bags.cuda(), masks.cuda()
            biom_embeds, biom_masks = biom_embeds.cuda(), biom_masks.cuda()
            metadata = metadata.cuda()

            optimizer.zero_grad()
            img_feat, txt_feat, logit_scale = model(bags, masks, biom_embeds, biom_masks, metadata)
            loss = contrastive_loss_ddp(img_feat, txt_feat, logit_scale)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

            if batch_idx % 20 == 0 and dist.get_rank() == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        scheduler.step()
        if dist.get_rank() == 0:
            save_checkpoint(
                model, optimizer, epoch, loss.item(),
                path=f"{config.model.checkpoint_dir}/checkpoint_epoch_{epoch}.pth",
            )
            logger.info(f"Epoch {epoch} complete. Avg loss: {running_loss / (batch_idx + 1):.4f}")


if __name__ == "__main__":
    main()
