"""Generate LLM-based biomarker embeddings using Sentence-BERT.

This script takes GPT-4o-generated text descriptions of biomarkers
(produced by prompting with the template in Fig. S1 of the paper)
and embeds them using the all-MiniLM-L6-v2 Sentence-BERT model
to produce fixed-length 384-dimensional embeddings.

Usage:
    python generate_llm_embeddings.py \
        --descriptions_pkl /path/to/llm_descriptions.pkl \
        --output_pkl /path/to/llm_sbert_embeddings.pkl

Input format:
    descriptions_pkl: dict {sample_id: [list of text descriptions]}
    Each text description is a GPT-4o output describing a biomarker's
    mutation and its role in the patient's cancer type.

Output format:
    dict {sample_id: np.ndarray of shape [n_biomarkers, 384]}
"""

import argparse
import pickle

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate SBERT embeddings from LLM descriptions")
    parser.add_argument("--descriptions_pkl", type=str, required=True, help="Path to GPT-4o descriptions pickle")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output path for SBERT embeddings")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence-BERT model name")
    args = parser.parse_args()

    with open(args.descriptions_pkl, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} samples")

    model = SentenceTransformer(args.model_name)

    results = {}
    for key, descriptions in tqdm(data.items(), desc="Embedding biomarkers"):
        embs = model.encode(descriptions, normalize_embeddings=True)
        results[key] = embs

    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved embeddings to {args.output_pkl}")


if __name__ == "__main__":
    main()
