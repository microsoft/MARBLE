"""Agata: Aggregator with Attention for tile embeddings.

Implements the Agata module (Raciti et al., 2023) for aggregating variable-length
sets of tile embeddings into a fixed-size patient-level representation using
learned cross-attention queries.
"""

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, NamedTuple

from torch import Tensor
from torch.nn import Linear, Module, Sequential


# ---------------------------------------------------------------------------
# FC Head utilities
# ---------------------------------------------------------------------------

class FCHeadOutput(NamedTuple):
    logits: Tensor
    activations: Tensor


@dataclass
class LinearLayerSpec:
    dim: int
    activation: Module


@dataclass
class FCHeadConfig:
    in_channels: int
    layer_specs: Sequence[LinearLayerSpec]


class FCHead(Module):
    def __init__(self, in_channels: int, layer_specs: Sequence[LinearLayerSpec]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.layer_specs = layer_specs

        ops: List[Module] = []
        for layer_spec in self.layer_specs[:-1]:
            ops.append(Linear(in_channels, layer_spec.dim))
            ops.append(layer_spec.activation)
            in_channels = layer_spec.dim
        ops.append(Linear(in_channels, self.layer_specs[-1].dim))

        self.activation = self.layer_specs[-1].activation
        self.fc = Sequential(*ops)

    def forward(self, x: Tensor) -> FCHeadOutput:
        logits = self.fc(x)
        activations = self.activation(logits)
        return FCHeadOutput(logits=logits, activations=activations)


# ---------------------------------------------------------------------------
# Dot-product attention with learned queries
# ---------------------------------------------------------------------------

class DotProductAttentionWithLearnedQueries(nn.Module):
    """Dot-product attention with a set of learned query vectors."""

    def __init__(
        self,
        in_features: int,
        n_queries: int = 1,
        scaled: bool = False,
        absolute: bool = False,
        reduce: bool = False,
        padding_indicator: float = 1,
        return_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.queries = nn.Linear(in_features, n_queries, bias=False)
        self.n_queries = n_queries
        self.scaled = scaled
        self.absolute = absolute
        self.reduce = reduce
        self.padding_indicator = padding_indicator
        self.return_normalized = return_normalized
        self.pad_value = 0

    def _format_padding_mask(
        self, padding_mask: Tensor, expected_shape: Tuple[int, int, int]
    ) -> Tensor:
        if padding_mask.shape[:2] != expected_shape[:2]:
            raise RuntimeError(
                f"Expected padding_mask shape {expected_shape[:2]} at first two dims, "
                f"got {padding_mask.shape[:2]}"
            )
        if len(padding_mask.shape) == 2:
            padding_mask = padding_mask.unsqueeze(-1)
        if len(padding_mask.shape) == 3:
            if padding_mask.shape[2] == 1:
                padding_mask = padding_mask.expand(-1, -1, expected_shape[2])
            elif padding_mask.shape[2] != expected_shape[2]:
                raise RuntimeError(
                    f"Expected padding_mask shape {expected_shape[:2]} or "
                    f"{(expected_shape[0], expected_shape[1], 1)}, got {padding_mask.shape}"
                )
        return padding_mask

    def forward(
        self, key: Tensor, value: Tensor, padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if key.size(1) != value.size(1):
            raise RuntimeError("Expected key and value to have same sequence length (dim 1).")

        attn = self.queries(key)  # (B, S, Q)

        if padding_mask is not None:
            padding_mask = self._format_padding_mask(
                padding_mask, expected_shape=(key.size(0), key.size(1), attn.size(2))
            )
            attn[padding_mask == self.padding_indicator] = self.pad_value

        if self.scaled:
            d_k = key.size(-1)
            attn /= math.sqrt(d_k)

        normalized_attn = F.softmax(attn, dim=1)

        if self.absolute:
            value = torch.abs(value)

        output = torch.bmm(normalized_attn.transpose(-2, -1), value)  # (B, Q, F)

        if self.reduce:
            output = output.sum(1)  # (B, F)

        if self.return_normalized:
            attn = normalized_attn

        return output, attn


# ---------------------------------------------------------------------------
# Agata aggregator
# ---------------------------------------------------------------------------

class Agata(nn.Module):
    """Two-layer feed-forward 'Aggregator with Attention' (Agata) model.

    Processes tile embeddings through two linear layers, then applies
    dot-product attention with learned queries to produce a fixed-size
    patient-level representation.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        layer1_out_features: int,
        layer2_out_features: int,
        activation: nn.Module,
        label_name_fclayer_head_config: Mapping[str, FCHeadConfig],
        scaled_attention: bool = False,
        absolute_attention: bool = False,
        n_attention_queries: int = 1,
        padding_indicator: int = 1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, layer1_out_features)
        self.linear2 = nn.Linear(layer1_out_features, layer2_out_features)

        self.attention = DotProductAttentionWithLearnedQueries(
            in_features=layer1_out_features,
            n_queries=n_attention_queries,
            scaled=scaled_attention,
            absolute=absolute_attention,
            reduce=True,
            padding_indicator=padding_indicator,
            return_normalized=False,
        )

        self.activation = activation

        heads = {
            name: FCHead(cfg.in_channels, cfg.layer_specs)
            for name, cfg in label_name_fclayer_head_config.items()
        }
        self.heads = cast(Mapping[str, FCHead], nn.ModuleDict(heads))
        self.class_head = nn.Sequential(
            nn.Linear(layer2_out_features, num_classes),
            nn.Sigmoid(),
        )

    def forward(
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Tensor:
        x_1, x_2 = self.forward_features(x)
        x_3, attn = self.attention(key=x_1, value=x_2, padding_mask=padding_masks)
        return x_3

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_1 = self.activation(self.linear1(x))
        x_2 = self.activation(self.linear2(x_1))
        return x_1, x_2

    def forward_with_heads(
        self, x: Tensor, padding_masks: Optional[Tensor]
    ) -> Dict[str, Tensor]:
        """Forward pass that also returns classification head outputs."""
        x_3 = self.forward(x, padding_masks)
        heads_logits = {}
        for name, head in self.heads.items():
            logits, activations = head(x_3)
            heads_logits[name] = activations
        return {"embedding": x_3, "heads_logits": self.class_head(x_3)}
