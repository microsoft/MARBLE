"""MARBLE model variants for contrastive pretraining and fine-tuning.

Implements:
- CrossAttentionAggregator: Biomarker embedding aggregator with multi-head cross-attention
- MARBLEModel_LLM: MARBLE-LLM (λ=1, image-LLM alignment only)
- MARBLEModel_LLM_PLM: MARBLE (λ=0.5) and MARBLE-PLM (λ=0) with joint/separate alignment
- MultihotEncoder: Multi-hot baseline biomarker encoder
- FinetuneModel: Supervised fine-tuning model with classification head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marble.modules import ProjectionHead
from marble.aggregator import Agata


class CrossAttentionAggregator(nn.Module):
    """Aggregates a variable-length set of biomarker embeddings into a single vector.

    Uses multi-head cross-attention with a single learnable query vector.
    Includes a learnable null token for patients with no biomarkers and
    a metadata projection for panel version / primary-metastatic status.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, use_mlp: bool = False):
        super().__init__()
        self.use_mlp = use_mlp

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, 128),
            )
            self.embed_dim_upd = 128
        else:
            self.embed_dim_upd = embed_dim

        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim_upd))
        self.attn = nn.MultiheadAttention(self.embed_dim_upd, num_heads, batch_first=True)
        self.null_token = nn.Parameter(torch.randn(self.embed_dim_upd))
        self.metadata_proj = nn.Linear(2, self.embed_dim_upd)

    def forward(self, embeddings, padding_mask, metadata_one_hot):
        if self.use_mlp:
            embeddings = self.mlp(embeddings)

        B = embeddings.size(0)

        # Append projected metadata token
        metadata_embed = self.metadata_proj(metadata_one_hot).unsqueeze(1)  # [B, 1, D]
        embeddings = torch.cat([embeddings, metadata_embed], dim=1)

        metadata_mask = torch.ones(B, 1, dtype=padding_mask.dtype, device=padding_mask.device)
        padding_mask = torch.cat([padding_mask, metadata_mask], dim=1)

        query = self.query.expand(B, -1, -1)

        # Handle empty biomarker sets with null token
        empty_bags = (~padding_mask).all(dim=1)
        if empty_bags.any():
            embeddings[empty_bags, 0, :] = self.null_token
            padding_mask[empty_bags, 0] = True

        attn_output, _ = self.attn(
            query, embeddings, embeddings, key_padding_mask=padding_mask.float()
        )
        return attn_output.squeeze(1)


class MultihotEncoder(nn.Module):
    """Multi-hot baseline biomarker encoder (following Vaidya et al., THREADS).

    Projects a binary multi-hot biomarker vector through a shallow MLP.
    """

    def __init__(self, embed_dim: int, use_mlp: bool = False):
        super().__init__()
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

    def forward(self, onehot_vector, metadata_one_hot):
        miss = onehot_vector == -999
        onehot_clean = torch.where(miss, onehot_vector.new_zeros(()), onehot_vector)
        if self.use_mlp:
            onehot_clean = self.mlp(onehot_clean)
        return onehot_clean.squeeze(1)


# ---------------------------------------------------------------------------
# MARBLE-LLM: single-modality contrastive model (λ = 1)
# ---------------------------------------------------------------------------

class MARBLEModel_LLM(nn.Module):
    """MARBLE-LLM: contrastive alignment between image and LLM biomarker embeddings."""

    def __init__(
        self,
        image_encoder: Agata,
        biom_encoder: CrossAttentionAggregator,
        image_embedding: int = 640,
        text_embedding: int = 384,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.biom_encoder = biom_encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, image_masks, text_features, text_masks, onehot_metadata):
        image_features = self.image_encoder(image_features, image_masks)
        biom_features = self.biom_encoder(text_features, text_masks, onehot_metadata)

        image_features = self.image_projection(image_features)
        text_features = self.text_projection(biom_features)

        # L2 normalize
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        logit_scale = self.logit_scale.exp()
        return image_features, text_features, logit_scale


# ---------------------------------------------------------------------------
# MARBLE: joint LLM + PLM contrastive model (λ ∈ {0, 0.5, 1})
# ---------------------------------------------------------------------------

class MARBLEModel_LLM_PLM(nn.Module):
    """MARBLE: joint contrastive alignment with LLM and PLM biomarker embeddings.

    When λ=0 → MARBLE-PLM, λ=0.5 → MARBLE, λ=1 → MARBLE-LLM.
    """

    def __init__(
        self,
        image_encoder: Agata,
        biom_encoder_plm: CrossAttentionAggregator,
        biom_encoder_llm: CrossAttentionAggregator,
        image_embedding: int = 640,
        text_embedding_llm: int = 384,
        text_embedding_plm: int = 1280,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.biom_encoder_plm = biom_encoder_plm
        self.biom_encoder_llm = biom_encoder_llm
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection_llm = ProjectionHead(embedding_dim=text_embedding_llm)
        self.text_projection_plm = ProjectionHead(embedding_dim=text_embedding_plm)
        self.logit_scale_llm = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_plm = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        image_features,
        image_masks,
        text_features_plm,
        text_masks_plm,
        text_features_llm,
        text_masks_llm,
        onehot_metadata,
    ):
        image_features = self.image_encoder(image_features, image_masks)

        biom_plm = self.biom_encoder_plm(text_features_plm, text_masks_plm, onehot_metadata)
        text_features_plm = self.text_projection_plm(biom_plm)

        biom_llm = self.biom_encoder_llm(text_features_llm, text_masks_llm, onehot_metadata)
        text_features_llm = self.text_projection_llm(biom_llm)

        image_features = self.image_projection(image_features)

        # L2 normalize all
        image_features = F.normalize(image_features, dim=1)
        text_features_plm = F.normalize(text_features_plm, dim=1)
        text_features_llm = F.normalize(text_features_llm, dim=1)

        logit_scale_plm = self.logit_scale_plm.exp()
        logit_scale_llm = self.logit_scale_llm.exp()

        return (
            image_features,
            text_features_plm,
            text_features_llm,
            logit_scale_plm,
            logit_scale_llm,
        )


# ---------------------------------------------------------------------------
# Multi-hot baseline contrastive model
# ---------------------------------------------------------------------------

class MARBLEModel_Multihot(nn.Module):
    """Multi-hot baseline: contrastive alignment with binary biomarker vectors."""

    def __init__(
        self,
        image_encoder: Agata,
        biom_encoder: MultihotEncoder,
        image_embedding: int = 640,
        vocab_size: int = 1600,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.biom_encoder = biom_encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, image_masks, onehot_label, onehot_metadata):
        image_features = self.image_encoder(image_features, image_masks)
        biom_features = self.biom_encoder(onehot_label, onehot_metadata)

        image_features = self.image_projection(image_features)
        text_features = self.text_projection(biom_features)

        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        logit_scale = self.logit_scale.exp()
        return image_features, text_features, logit_scale


# ---------------------------------------------------------------------------
# Fine-tuning model
# ---------------------------------------------------------------------------

class FinetuneModel(nn.Module):
    """Supervised fine-tuning model.

    Initializes the image aggregator + projection head from MARBLE pretraining,
    then adds a multi-label classification head.
    """

    def __init__(
        self,
        image_encoder: Agata,
        image_embedding: int = 640,
        num_classes: int = 1611,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.class_head = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, image_features, image_masks):
        image_features = self.image_encoder(image_features, image_masks)
        image_features = self.image_projection(image_features)
        return self.class_head(image_features)
