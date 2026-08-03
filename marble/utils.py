"""Utility functions for training, evaluation, and data loading."""

import os
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file, casting dictionary columns to strings."""
    table = pq.read_table(path)
    schema = table.schema
    new_fields = [
        pa.field(name, pa.string()) if pa.types.is_dictionary(field.type) else field
        for name, field in zip(schema.names, schema)
    ]
    table = table.cast(pa.schema(new_fields))
    return table.to_pandas()


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    """Save model and optimizer state."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False):
    """Load pretrained weights into a model, handling DDP state dict prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Remove 'module.' prefix from DDP checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=strict)
    return model


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute balanced accuracy for a single biomarker."""
    y_bin = (y_pred >= threshold).astype(int)
    tp = ((y_bin == 1) & (y_true == 1)).sum()
    tn = ((y_bin == 0) & (y_true == 0)).sum()
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    sensitivity = tp / pos if pos > 0 else 0.0
    specificity = tn / neg if neg > 0 else 0.0
    return (sensitivity + specificity) / 2


def compute_auc_per_class(all_labels: np.ndarray, all_preds: np.ndarray, mask_value: float = -999) -> np.ndarray:
    """Compute per-biomarker AUC, skipping masked or single-class columns."""
    num_classes = all_labels.shape[1]
    auc_scores = []
    for i in range(num_classes):
        mask = all_labels[:, i] != mask_value
        if np.sum(mask) < 2 or len(np.unique(all_labels[mask, i])) < 2:
            auc_scores.append(np.nan)
            continue
        try:
            auc = roc_auc_score(all_labels[mask, i], all_preds[mask, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(np.nan)
    return np.array(auc_scores)


def load_data_panel_version(config, mode="train"):
    """Load and filter patient data for a given panel version configuration.

    Applies exclusion lists, QC filters, and panel version filtering.
    Returns the filtered dataframe and biomarker panel indices.
    """
    if mode == "tune":
        train_df = read_parquet(config.dataset.train_parquet.replace("train", "tune"))
    else:
        train_df = read_parquet(config.dataset.train_parquet)

    # Exclude bad slides
    excl = pd.read_csv(config.dataset.excl_list, header=None)
    train_df = train_df[~train_df["image_file_name"].isin(excl[0])]

    # Exclude TCGA samples
    train_df = train_df[train_df["dmp_sample_key"] != "-999"]

    # Exclude QC-failed patients
    failed_patients = np.load(config.dataset.drop_patients, allow_pickle=True).tolist()
    failed_patients += np.load(
        config.dataset.drop_patients.replace("train", "tune"), allow_pickle=True
    ).tolist()
    train_df = train_df[~train_df["patient_hid"].isin(failed_patients)]

    # Filter to patients with existing bag files
    def bag_file_exists(bag_id):
        bag_name = f"{str(bag_id).split('.svs')[0]}.asdf"
        bag_dir = config.dataset.path_to_bags
        if mode == "tune":
            bag_dir = bag_dir.replace("train", "tune")
        return os.path.exists(os.path.join(bag_dir, bag_name))

    train_df = train_df[train_df["image_file_name"].apply(bag_file_exists)]

    # Filter to pretraining panel versions (v3, v5, v6)
    impact_version = read_parquet(config.dataset.impact_version)
    versions_to_train = ["IMPACT341", "IMPACT410", "IMPACT468"]
    impact_samples = []
    for item in versions_to_train:
        impact_samples.extend(
            impact_version.loc[impact_version["panel_name"] == item, "msk_dmp_sample_id"].tolist()
        )
    train_df = train_df[train_df["msk_dmp_sample_id"].isin(impact_samples)]

    # Get biomarker panel indices
    panel_label_mapping = pd.read_csv(config.dataset.panel_label_mapping)
    biomarker_names = np.load(config.dataset.biomarker_names, allow_pickle=True).tolist()
    biom_panel = panel_label_mapping[
        ~panel_label_mapping["internal_label"].isin(biomarker_names)
    ].index.tolist()

    indices_to_drop = np.load(config.dataset.drop_biomarkers)
    biom_panel = [idx for idx in biom_panel if idx not in indices_to_drop]

    return train_df, biom_panel, impact_version
