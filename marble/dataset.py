"""Dataset classes for MARBLE pretraining and fine-tuning.

Provides PyTorch Dataset implementations for:
- PretrainDataset_LLM: LLM-only contrastive pretraining
- PretrainDataset_LLM_PLM: Joint LLM+PLM contrastive pretraining
- FinetuneDataset: Supervised fine-tuning for biomarker classification
"""

import os
import random

import asdf
import numpy as np
import torch
from torch.utils.data import Dataset

PANEL_NAMES = ["IMPACT341", "IMPACT410", "IMPACT468"]
SAMPLE_TYPES = ["Primary", "Metastasis"]
MAX_TILES = 25000
MAX_BIOMARKERS = 30


def one_hot_encode(panel: str, sample: str) -> np.ndarray:
    """Encode panel version and sample type as a 2D metadata vector."""
    panel_index = PANEL_NAMES.index(panel)
    sample_index = SAMPLE_TYPES.index(sample)
    return np.array([panel_index, sample_index])


def _load_bag(path_to_bags: str, bag_id: str) -> np.ndarray:
    """Load tile embeddings from an ASDF file."""
    bag_path = os.path.join(path_to_bags, f"{str(bag_id).split('.svs')[0]}.asdf")
    if not os.path.exists(bag_path) or not bag_path.endswith(".asdf"):
        return np.zeros((0, 128))
    try:
        with asdf.open(bag_path) as f:
            return f["embeddings"]["data"].__array__().squeeze(1)
    except Exception:
        return np.zeros((0, 128))


def _pad_bag(bag: np.ndarray, max_len: int):
    """Pad or subsample a bag of tile embeddings to a fixed length."""
    length = min(len(bag), max_len)
    padded = np.zeros((max_len, bag.shape[1]), dtype=bag.dtype)
    if len(bag) > max_len:
        indices = random.sample(range(len(bag)), max_len)
        padded[:length] = bag[indices]
    else:
        padded[:length] = bag[:length]
    mask = np.zeros(max_len, dtype=bool)
    mask[:length] = True
    return padded, mask


def _pad_embeddings(embs: np.ndarray, max_len: int, embed_dim: int):
    """Pad biomarker embeddings to a fixed length."""
    if len(embs) > max_len:
        indices = random.sample(range(len(embs)), max_len)
        padded = np.zeros((max_len, embs.shape[1]), dtype=embs.dtype)
        padded[:max_len] = embs[indices]
    elif len(embs) > 0:
        padded = np.zeros((max_len, embs.shape[1]), dtype=embs.dtype)
        padded[: len(embs)] = embs[:max_len]
    else:
        padded = np.zeros((max_len, embed_dim), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    mask[: min(len(embs), max_len)] = True
    return padded, mask


# ---------------------------------------------------------------------------
# LLM-only pretraining dataset
# ---------------------------------------------------------------------------

class PretrainDataset_LLM(Dataset):
    """Dataset for MARBLE-LLM contrastive pretraining.

    Returns tile embeddings, LLM-based biomarker embeddings, and metadata.
    """

    def __init__(
        self, df, biom_panel, path_to_bags, embeds_llm,
        descriptions, after_first_word, gene_names, impact_version,
    ):
        self.df = df
        self.biom_panel = biom_panel
        self.patients = list(df["patient_hid"].unique())
        self.path_to_bags = path_to_bags
        self.embeds_llm = embeds_llm
        self.descriptions = descriptions
        self.after_first_word = after_first_word
        self.gene_names = gene_names
        self.impact_version = impact_version

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        tmp_df = self.df[self.df["patient_hid"] == self.patients[idx]]
        msk_id = tmp_df.iloc[0]["msk_dmp_sample_id"]
        panel = self.impact_version[
            self.impact_version["msk_dmp_sample_id"] == msk_id
        ]["panel_name"].values[0]
        sample = tmp_df.iloc[0]["primary_or_metastasis"]
        metadata = one_hot_encode(panel, sample)

        bag, embs_llm = [], []
        for i in range(tmp_df.shape[0]):
            bag.extend(_load_bag(self.path_to_bags, tmp_df.iloc[i]["image_file_name"]))
            if i == 0:
                embs_llm = self._get_llm_embeddings(tmp_df.iloc[i])

        # Fallback for empty bags
        j = 0
        while len(bag) == 0:
            fallback_df = self.df[self.df["patient_hid"] == self.patients[j]]
            bag = []
            for i in range(fallback_df.shape[0]):
                bag.extend(_load_bag(self.path_to_bags, fallback_df.iloc[i]["image_file_name"]))
            j += 1

        label = np.array(tmp_df.iloc[0]["omniscreen"][self.biom_panel])
        bag = np.array(bag)
        padded_bag, mask = _pad_bag(bag, MAX_TILES)

        embs_llm = np.array(embs_llm) if len(embs_llm) > 0 else np.array(embs_llm)
        padded_llm, mask_llm = _pad_embeddings(embs_llm, MAX_BIOMARKERS, 384)

        return (
            torch.from_numpy(padded_bag).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(padded_llm).float(),
            torch.from_numpy(mask_llm).bool(),
            torch.from_numpy(metadata).float(),
        )

    def _get_llm_embeddings(self, row):
        """Extract LLM embeddings for positive biomarkers of a patient."""
        cancer = row["detailed_tumor_type"]
        panel = row["omniscreen"][self.biom_panel]
        idxs = np.where(panel == 1)[0]
        embs = []
        for idx in idxs:
            tmp = self.descriptions.iloc[idx]
            name = tmp.split(" ")[0].upper()
            if name in self.gene_names:
                gene, mutation = name, self.after_first_word.iloc[idx]
            else:
                gene, mutation = "", tmp
            key = f"{gene}-{mutation}-{cancer}"
            if key in self.embeds_llm:
                embs.append(self.embeds_llm[key])
        return embs


# ---------------------------------------------------------------------------
# PLM-only pretraining dataset
# ---------------------------------------------------------------------------

class PretrainDataset_PLM(Dataset):
    """Dataset for MARBLE-PLM contrastive pretraining.

    Returns tile embeddings, ESM-2-based biomarker embeddings, and metadata.
    """

    def __init__(
        self, df, biom_panel, path_to_bags, embeds_esm2,
        descriptions, after_first_word, gene_names, impact_version,
    ):
        self.df = df
        self.biom_panel = biom_panel
        self.patients = list(df["patient_hid"].unique())
        self.path_to_bags = path_to_bags
        self.embeds_esm2 = embeds_esm2
        self.descriptions = descriptions
        self.after_first_word = after_first_word
        self.gene_names = gene_names
        self.impact_version = impact_version

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        tmp_df = self.df[self.df["patient_hid"] == self.patients[idx]]
        msk_id = tmp_df.iloc[0]["msk_dmp_sample_id"]
        panel = self.impact_version[
            self.impact_version["msk_dmp_sample_id"] == msk_id
        ]["panel_name"].values[0]
        sample = tmp_df.iloc[0]["primary_or_metastasis"]
        metadata = one_hot_encode(panel, sample)

        bag, embs_plm = [], []
        for i in range(tmp_df.shape[0]):
            bag.extend(_load_bag(self.path_to_bags, tmp_df.iloc[i]["image_file_name"]))

            # PLM embeddings from ESM-2
            list_esm = self.embeds_esm2.get(msk_id, [])
            for item in list_esm:
                if item is not None and len(item) == 1280:
                    embs_plm.append(np.array(item))

        # Fallback for empty bags
        j = 0
        while len(bag) == 0:
            fallback_df = self.df[self.df["patient_hid"] == self.patients[j]]
            bag = []
            for i in range(fallback_df.shape[0]):
                bag.extend(_load_bag(self.path_to_bags, fallback_df.iloc[i]["image_file_name"]))
            j += 1

        label = np.array(tmp_df.iloc[0]["omniscreen"][self.biom_panel])
        bag = np.array(bag)
        padded_bag, mask = _pad_bag(bag, MAX_TILES)

        embs_plm = np.array(embs_plm) if len(embs_plm) > 0 else np.array(embs_plm)
        padded_plm, mask_plm = _pad_embeddings(embs_plm, MAX_BIOMARKERS, 1280)

        return (
            torch.from_numpy(padded_bag).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(padded_plm).float(),
            torch.from_numpy(mask_plm).bool(),
            torch.from_numpy(metadata).float(),
        )


# ---------------------------------------------------------------------------
# Joint LLM + PLM pretraining dataset
# ---------------------------------------------------------------------------

class PretrainDataset_LLM_PLM(Dataset):
    """Dataset for MARBLE joint LLM+PLM contrastive pretraining.

    Returns tile embeddings, both LLM and PLM biomarker embeddings, and metadata.
    """

    def __init__(
        self, df, biom_panel, path_to_bags, embeds_esm2, embeds_llm,
        descriptions, after_first_word, gene_names, impact_version,
    ):
        self.df = df
        self.biom_panel = biom_panel
        self.patients = list(df["patient_hid"].unique())
        self.path_to_bags = path_to_bags
        self.embeds_llm = embeds_llm
        self.embeds_esm2 = embeds_esm2
        self.descriptions = descriptions
        self.after_first_word = after_first_word
        self.gene_names = gene_names
        self.impact_version = impact_version

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        tmp_df = self.df[self.df["patient_hid"] == self.patients[idx]]
        msk_id = tmp_df.iloc[0]["msk_dmp_sample_id"]
        panel = self.impact_version[
            self.impact_version["msk_dmp_sample_id"] == msk_id
        ]["panel_name"].values[0]
        sample = tmp_df.iloc[0]["primary_or_metastasis"]
        metadata = one_hot_encode(panel, sample)

        bag, embs_plm, embs_llm = [], [], []
        for i in range(tmp_df.shape[0]):
            bag.extend(_load_bag(self.path_to_bags, tmp_df.iloc[i]["image_file_name"]))

            # PLM embeddings from ESM-2
            list_esm = self.embeds_esm2.get(msk_id, [])
            for item in list_esm:
                if item is not None and len(item) == 1280:
                    embs_plm.append(np.array(item))

            # LLM embeddings
            if i == 0:
                embs_llm = self._get_llm_embeddings(tmp_df.iloc[i])

        # Fallback for empty bags
        j = 0
        while len(bag) == 0:
            fallback_df = self.df[self.df["patient_hid"] == self.patients[j]]
            bag = []
            for i in range(fallback_df.shape[0]):
                bag.extend(_load_bag(self.path_to_bags, fallback_df.iloc[i]["image_file_name"]))
            j += 1

        label = np.array(tmp_df.iloc[0]["omniscreen"][self.biom_panel])
        bag = np.array(bag)
        padded_bag, mask = _pad_bag(bag, MAX_TILES)

        embs_plm = np.array(embs_plm) if len(embs_plm) > 0 else np.array(embs_plm)
        padded_plm, mask_plm = _pad_embeddings(embs_plm, MAX_BIOMARKERS, 1280)

        embs_llm = np.array(embs_llm) if len(embs_llm) > 0 else np.array(embs_llm)
        padded_llm, mask_llm = _pad_embeddings(embs_llm, MAX_BIOMARKERS, 384)

        return (
            torch.from_numpy(padded_bag).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(padded_plm).float(),
            torch.from_numpy(mask_plm).bool(),
            torch.from_numpy(padded_llm).float(),
            torch.from_numpy(mask_llm).bool(),
            torch.from_numpy(metadata).float(),
        )

    def _get_llm_embeddings(self, row):
        cancer = row["detailed_tumor_type"]
        panel = row["omniscreen"][self.biom_panel]
        idxs = np.where(panel == 1)[0]
        embs = []
        for idx in idxs:
            tmp = self.descriptions.iloc[idx]
            name = tmp.split(" ")[0].upper()
            if name in self.gene_names:
                gene, mutation = name, self.after_first_word.iloc[idx]
            else:
                gene, mutation = "", tmp
            key = f"{gene}-{mutation}-{cancer}"
            if key in self.embeds_llm:
                embs.append(self.embeds_llm[key])
        return embs


# ---------------------------------------------------------------------------
# Fine-tuning dataset
# ---------------------------------------------------------------------------

class FinetuneDataset(Dataset):
    """Dataset for supervised fine-tuning on multi-label biomarker classification."""

    def __init__(self, df, biom_panel, path_to_bags, max_len=25000):
        self.df = df
        self.biom_panel = biom_panel
        self.patients = list(df["patient_hid"].unique())
        self.path_to_bags = path_to_bags
        self.max_len = max_len

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        tmp_df = self.df[self.df["patient_hid"] == self.patients[idx]]
        bag = []
        for i in range(tmp_df.shape[0]):
            bag.extend(_load_bag(self.path_to_bags, tmp_df.iloc[i]["image_file_name"]))

        label = tmp_df.iloc[0]["omniscreen"][self.biom_panel]

        if len(bag) == 0:
            fallback_df = self.df[self.df["patient_hid"] == self.patients[0]]
            bag = []
            for i in range(fallback_df.shape[0]):
                bag.extend(_load_bag(self.path_to_bags, fallback_df.iloc[i]["image_file_name"]))
            label = fallback_df.iloc[0]["omniscreen"][self.biom_panel]

        bag = np.array(bag)
        label = np.array(label)
        padded_bag, mask = _pad_bag(bag, self.max_len)

        return (
            torch.from_numpy(padded_bag).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(label).float(),
        )
