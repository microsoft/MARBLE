"""MARBLE joint LLM+PLM contrastive pretraining (Scenario 2).

Usage:
    torchrun --nproc_per_node=4 train_pretrain_llm_plm.py --config configs/pretrain_llm_plm.yaml
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
from marble.dataset import PretrainDataset_LLM_PLM
from marble.loss import contrastive_loss_ddp
from marble.model import CrossAttentionAggregator, MARBLEModel_LLM_PLM
from marble.utils import save_checkpoint, set_seed, read_parquet


def parse_args():
    parser = argparse.ArgumentParser(description="MARBLE LLM+PLM contrastive pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--lambda_weight", type=float, default=0.5, help="Weight for LLM loss (0=PLM only, 0.5=joint, 1=LLM only)")
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


def load_data_plm_experiment(config, mode="train"):
    """Load data for PLM experiment (v3 + v5 panels only)."""
    embeds_esm2 = np.load(config.dataset.embs_esm2, allow_pickle=True).item()
    ids = list(embeds_esm2.keys())

    if mode == "tune":
        df = read_parquet(config.dataset.train_parquet.replace("train", "tune"))
    else:
        df = read_parquet(config.dataset.train_parquet)

    df = df[df["msk_dmp_sample_id"].isin(ids)]

    excl = pd.read_csv(config.dataset.excl_list, header=None)
    df = df[~df["image_file_name"].isin(excl[0])]
    df = df[df["dmp_sample_key"] != "-999"]

    failed = np.load(config.dataset.drop_patients, allow_pickle=True).tolist()
    failed += np.load(config.dataset.drop_patients.replace("train", "tune"), allow_pickle=True).tolist()
    df = df[~df["patient_hid"].isin(failed)]

    def bag_exists(bag_id):
        bag_dir = config.dataset.path_to_bags
        if mode == "tune":
            bag_dir = bag_dir.replace("train", "tune")
        return os.path.exists(os.path.join(bag_dir, f"{str(bag_id).split('.svs')[0]}.asdf"))

    df = df[df["image_file_name"].apply(bag_exists)]

    impact_version = read_parquet(config.dataset.impact_version)
    versions = ["IMPACT341", "IMPACT410"]
    samples = []
    for v in versions:
        samples.extend(impact_version.loc[impact_version["panel_name"] == v, "msk_dmp_sample_id"].tolist())
    df = df[df["msk_dmp_sample_id"].isin(samples)]

    panel_label_mapping = pd.read_csv(config.dataset.panel_label_mapping)
    biomarker_names = np.load(config.dataset.biomarker_names, allow_pickle=True).tolist()
    biom_panel = panel_label_mapping[~panel_label_mapping["internal_label"].isin(biomarker_names)].index.tolist()
    indices_to_drop = np.load(config.dataset.drop_biomarkers)
    biom_panel = [idx for idx in biom_panel if idx not in indices_to_drop]

    return df, biom_panel, impact_version


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    lam = args.lambda_weight
    set_seed(42)

    train_df, biom_panel, impact_version = load_data_plm_experiment(config)

    panel_label_mapping = pd.read_csv(config.dataset.panel_label_mapping)
    descriptions = panel_label_mapping["biomarker_description"]
    after_first_word = descriptions.str.split(" ", n=1).str[1]
    cgl = pd.read_csv(config.dataset.cancer_gene_list, sep="\t", header=0)
    gene_names = cgl["Hugo Symbol"].tolist()

    embeds_esm2 = np.load(config.dataset.embs_esm2, allow_pickle=True).item()

    local_rank = setup_ddp()

    with open(config.dataset.embeds_tumor_gene_path, "rb") as f:
        embeds_llm = pickle.load(f)

    train_dataset = PretrainDataset_LLM_PLM(
        train_df, biom_panel, config.dataset.path_to_bags,
        embeds_esm2, embeds_llm, descriptions, after_first_word, gene_names, impact_version,
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

    biom_encoder_plm = CrossAttentionAggregator(1280).cuda()
    biom_encoder_llm = CrossAttentionAggregator(384).cuda()

    model = MARBLEModel_LLM_PLM(
        image_encoder=image_agg,
        biom_encoder_plm=biom_encoder_plm,
        biom_encoder_llm=biom_encoder_llm,
        text_embedding_llm=384,
        text_embedding_plm=1280,
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)

    os.makedirs(config.model.checkpoint_dir, exist_ok=True)
    os.makedirs(config.model.log_dir, exist_ok=True)
    log_file = Path(config.model.log_dir) / "training_log.txt"
    logging.basicConfig(filename=log_file, filemode="a", format="%(asctime)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger()

    for epoch in range(config.training.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.get_rank() != 0))
        for batch_idx, (bags, masks, labels, plm_embeds, plm_masks, llm_embeds, llm_masks, metadata) in enumerate(pbar):
            bags, masks = bags.cuda(), masks.cuda()
            plm_embeds, plm_masks = plm_embeds.cuda(), plm_masks.cuda()
            llm_embeds, llm_masks = llm_embeds.cuda(), llm_masks.cuda()
            metadata = metadata.cuda()

            optimizer.zero_grad()
            img_feat, plm_feat, llm_feat, scale_plm, scale_llm = model(
                bags, masks, plm_embeds, plm_masks, llm_embeds, llm_masks, metadata,
            )

            loss_plm = contrastive_loss_ddp(img_feat, plm_feat, scale_plm)
            loss_llm = contrastive_loss_ddp(img_feat, llm_feat, scale_llm)
            # L_total = λ * L_im-LLM + (1-λ) * L_im-PLM  (Eq. 4)
            loss = lam * loss_llm + (1 - lam) * loss_plm

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
