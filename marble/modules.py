"""Projection head for contrastive learning."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with residual connection and layer normalization.

    Maps aggregated embeddings to a common d-dimensional space for contrastive alignment.
    """

    def __init__(self, embedding_dim: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # residual connection
        x = self.layer_norm(x)
        return x
