"""Loss functions for MARBLE pretraining and fine-tuning."""

import torch
import torch.nn.functional as F
import torch.distributed as dist


def contrastive_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss (Eq. 3 in the paper).

    Args:
        image_features: L2-normalized image embeddings [B, D].
        text_features: L2-normalized text embeddings [B, D].
        logit_scale: Learnable temperature parameter (exp of log-scale).

    Returns:
        Scalar contrastive loss.
    """
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2


def contrastive_loss_ddp(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE loss with DDP feature gathering across GPUs.

    Gathers features from all ranks so that the contrastive loss is computed
    over the global batch, while gradients flow only through the local shard.
    """
    image_features_all = _gather_features_ddp(image_features)
    text_features_all = _gather_features_ddp(text_features)

    logits_per_image = logit_scale * image_features @ text_features_all.t()
    logits_per_text = logit_scale * text_features @ image_features_all.t()

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_batch_size = image_features.shape[0]
    start_index = rank * local_batch_size
    targets = torch.arange(start_index, start_index + local_batch_size, device=image_features.device)

    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    return (loss_i2t + loss_t2i) / 2


def masked_bce_loss(predictions: torch.Tensor, targets: torch.Tensor, mask_value: float = -999) -> torch.Tensor:
    """Binary cross-entropy loss that ignores masked biomarker labels.

    Biomarkers not measured for a given patient are marked with `mask_value`
    and excluded from the loss computation.
    """
    mask = targets != mask_value
    return F.binary_cross_entropy(predictions[mask].float(), targets[mask].float(), reduction="mean")


def _gather_features_ddp(features: torch.Tensor) -> torch.Tensor:
    """Gather features from all DDP ranks."""
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered, features)
        return torch.cat(gathered, dim=0)
    return features
