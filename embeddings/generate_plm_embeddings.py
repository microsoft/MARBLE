"""Generate PLM-based biomarker embeddings using ESM-2.

This script:
1. Reads mutation data (HGVS protein annotations) from a TSV file
2. Fetches canonical protein sequences from UniProt
3. Applies mutations to generate altered protein sequences
4. Embeds the mutated sequences using ESM-2 (650M parameter model)

Usage:
    python generate_plm_embeddings.py \
        --mutations_file /path/to/data_mutations.txt \
        --output_npy /path/to/esm2_embeddings.npy

Input format:
    Tab-separated file with columns including:
    - Tumor_Sample_Barcode: sample identifier
    - Hugo_Symbol: gene name
    - HGVSp_Short: HGVS protein notation (e.g., p.V600E)

Output format:
    dict {sample_id: [list of 1280-dim embeddings or None per mutation]}
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from protein_editing import edit_protein, parse_fasta, uniprot_fasta_by_gene


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings for mutated proteins")
    parser.add_argument("--mutations_file", type=str, required=True, help="Path to mutations TSV file")
    parser.add_argument("--output_npy", type=str, required=True, help="Output path for ESM-2 embeddings")
    parser.add_argument("--tokens_per_batch", type=int, default=10000, help="Token budget per batch")
    args = parser.parse_args()

    # Step 1: Generate mutated amino acid sequences
    print("Step 1: Generating mutated protein sequences...")
    df = pd.read_csv(args.mutations_file, sep="\t", low_memory=False)

    res_mutated_aas = defaultdict(list)
    for i in tqdm(range(df.shape[0]), desc="Processing mutations"):
        try:
            sample_id = df.iloc[i]["Tumor_Sample_Barcode"]
            hugo = df.iloc[i]["Hugo_Symbol"]
            mut = df.iloc[i]["HGVSp_Short"]
            fasta = uniprot_fasta_by_gene(hugo, reviewed=True)
            for acc, hdr, seq in parse_fasta(fasta):
                final_seq = seq
            r = edit_protein(final_seq, mut, strict_wt=False)
            res_mutated_aas[sample_id].append(r["new_seq"])
        except Exception:
            res_mutated_aas[sample_id].append(None)

    # Step 2: Embed with ESM-2
    print("Step 2: Embedding with ESM-2...")
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Deduplicate sequences
    seq_to_ids = defaultdict(list)
    for sample_id, seqs in res_mutated_aas.items():
        for i, seq in enumerate(seqs):
            if seq:
                seq_to_ids[seq].append((sample_id, i))

    unique_seqs = list(seq_to_ids.keys())
    seq_lens = [len(s) for s in unique_seqs]
    sorted_idx = sorted(range(len(unique_seqs)), key=lambda i: seq_lens[i])
    unique_seqs = [unique_seqs[i] for i in sorted_idx]
    seq_lens = [seq_lens[i] for i in sorted_idx]

    # Batch by token budget
    batches = []
    cur, cur_tokens = [], 0
    for s, L in zip(unique_seqs, seq_lens):
        if cur and (cur_tokens + L) > args.tokens_per_batch:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(s)
        cur_tokens += L
    if cur:
        batches.append(cur)

    emb_cache = {}

    with torch.inference_mode():
        for batch in tqdm(batches, desc="ESM-2 embedding"):
            data = [(f"p{i}", s) for i, s in enumerate(batch)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            
            out = model(batch_tokens, repr_layers=[33], return_contacts=False)
            toks_repr = out["representations"][33]

            B, T, D = toks_repr.shape
            arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            end = (batch_lens - 1).unsqueeze(1)
            mask = (arange >= 1) & (arange < end)
            masked = toks_repr * mask.unsqueeze(-1)
            sums = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            seq_embeds = (sums / counts).cpu()

            for s, e in zip(batch, seq_embeds):
                emb_cache[s] = e.numpy()

    # Map back to samples
    sample_to_embs = {}
    for sample_id, seqs in res_mutated_aas.items():
        embs = [None if s is None else emb_cache.get(s) for s in seqs]
        sample_to_embs[sample_id] = embs

    np.save(args.output_npy, sample_to_embs)
    print(f"Saved ESM-2 embeddings for {len(sample_to_embs)} samples to {args.output_npy}")


if __name__ == "__main__":
    main()
