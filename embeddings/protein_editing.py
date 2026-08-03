"""HGVS protein sequence editing utilities.

Applies in-frame protein-level HGVS variants to canonical UniProt sequences.
Supports: missense, nonsense, single-AA deletion/duplication, range duplication,
adjacent insertion, and in-frame delins mutations.

Used to construct mutated protein sequences for ESM-2 embedding.
"""

import re
import requests
from typing import Dict, List, Optional, Tuple

AA = set("ACDEFGHIKLMNPQRSTVWY")

# Regex patterns for supported HGVS protein variants
_missense = re.compile(r"^p\.?([A-Z])(\d+)([A-Z])$")
_nonsense = re.compile(r"^p\.?([A-Z])(\d+)\*$")
_del_one = re.compile(r"^p\.?([A-Z])(\d+)del$")
_dup_one = re.compile(r"^p\.?([A-Z])(\d+)dup$")
_dup_rng = re.compile(r"^p\.?([A-Z])(\d+)_([A-Z])(\d+)dup$")
_ins_adj = re.compile(r"^p\.?([A-Z])(\d+)_([A-Z])(\d+)ins([A-Z]+)$")
_delins_rng = re.compile(r"^p\.?([A-Z])(\d+)_([A-Z])(\d+)delins([A-Z]+)$")


def _check_pos(seq: str, pos: int):
    if not (1 <= pos <= len(seq)):
        raise ValueError(f"Position {pos} out of bounds for sequence length {len(seq)}")


def _all_aa(s: str) -> bool:
    return all(c in AA for c in s)


def edit_protein(seq: str, hgvs: str, strict_wt: bool = False) -> Dict:
    """Apply an HGVS protein edit to a sequence.

    Args:
        seq: Canonical protein sequence.
        hgvs: HGVS protein notation (e.g., 'p.V600E').
        strict_wt: If True, verify wild-type residue matches.

    Returns:
        Dict with keys: ok (bool), type (str), new_seq (str), reason (str).
    """
    s = hgvs.strip()

    try:
        # Missense p.X123Y
        m = _missense.match(s)
        if m:
            wt, pos, new = m.groups()
            pos = int(pos)
            _check_pos(seq, pos)
            if strict_wt and seq[pos - 1] != wt:
                return {"ok": False, "type": None, "new_seq": None, "reason": f"WT mismatch at {pos}"}
            return {"ok": True, "type": "missense", "new_seq": seq[: pos - 1] + new + seq[pos:], "reason": None}

        # Nonsense p.X123*
        m = _nonsense.match(s)
        if m:
            wt, pos = m.groups()
            pos = int(pos)
            _check_pos(seq, pos)
            if strict_wt and seq[pos - 1] != wt:
                return {"ok": False, "type": None, "new_seq": None, "reason": f"WT mismatch at {pos}"}
            return {"ok": True, "type": "nonsense", "new_seq": seq[: pos - 1], "reason": None}

        # Single AA deletion p.X123del
        m = _del_one.match(s)
        if m:
            wt, pos = m.groups()
            pos = int(pos)
            _check_pos(seq, pos)
            if strict_wt and seq[pos - 1] != wt:
                return {"ok": False, "type": None, "new_seq": None, "reason": f"WT mismatch at {pos}"}
            return {"ok": True, "type": "deletion_one", "new_seq": seq[: pos - 1] + seq[pos:], "reason": None}

        # Single AA duplication p.X123dup
        m = _dup_one.match(s)
        if m:
            wt, pos = m.groups()
            pos = int(pos)
            _check_pos(seq, pos)
            if strict_wt and seq[pos - 1] != wt:
                return {"ok": False, "type": None, "new_seq": None, "reason": f"WT mismatch at {pos}"}
            return {"ok": True, "type": "duplication_one", "new_seq": seq[:pos] + seq[pos - 1] + seq[pos:], "reason": None}

        # Range duplication p.X123_Y130dup
        m = _dup_rng.match(s)
        if m:
            lw, lp, rw, rp = m.groups()
            lp, rp = int(lp), int(rp)
            if lp > rp:
                return {"ok": False, "type": None, "new_seq": None, "reason": "Invalid range"}
            _check_pos(seq, lp)
            _check_pos(seq, rp)
            frag = seq[lp - 1 : rp]
            return {"ok": True, "type": "duplication_range", "new_seq": seq[:rp] + frag + seq[rp:], "reason": None}

        # Adjacent insertion p.X123_X124insABC
        m = _ins_adj.match(s)
        if m:
            lw, lp, rw, rp, ins = m.groups()
            lp, rp = int(lp), int(rp)
            if rp != lp + 1:
                return {"ok": False, "type": None, "new_seq": None, "reason": "Insertion not adjacent"}
            _check_pos(seq, lp)
            _check_pos(seq, rp)
            if not _all_aa(ins):
                return {"ok": False, "type": None, "new_seq": None, "reason": "Non-standard AA in insertion"}
            return {"ok": True, "type": "insertion_adjacent", "new_seq": seq[:lp] + ins + seq[lp:], "reason": None}

        # In-frame delins p.X123_Y130delinsAB
        m = _delins_rng.match(s)
        if m:
            lw, lp, rw, rp, ins = m.groups()
            lp, rp = int(lp), int(rp)
            if lp > rp:
                return {"ok": False, "type": None, "new_seq": None, "reason": "Invalid range"}
            if not _all_aa(ins):
                return {"ok": False, "type": None, "new_seq": None, "reason": "Non-standard AA in delins"}
            _check_pos(seq, lp)
            _check_pos(seq, rp)
            return {"ok": True, "type": "delins", "new_seq": seq[: lp - 1] + ins + seq[rp:], "reason": None}

        return {"ok": False, "type": None, "new_seq": None, "reason": "Unsupported HGVS variant"}

    except Exception as e:
        return {"ok": False, "type": None, "new_seq": None, "reason": f"Error: {e}"}


def uniprot_fasta_by_gene(gene_symbol: str, organism_id: int = 9606, reviewed: bool = True, timeout: int = 30) -> str:
    """Fetch FASTA sequence from UniProt for a given gene symbol."""
    query_parts = [f"gene_exact:{gene_symbol}", f"organism_id:{organism_id}"]
    if reviewed:
        query_parts.append("reviewed:true")
    query = " AND ".join(query_parts)
    url = "https://rest.uniprot.org/uniprotkb/stream"
    r = requests.get(url, params={"query": query, "format": "fasta"}, timeout=timeout)
    r.raise_for_status()
    if not r.text.strip():
        raise ValueError(f"No UniProt sequences found for {gene_symbol}.")
    return r.text


def parse_fasta(fasta_text: str):
    """Yield (accession, header, sequence) tuples from FASTA text."""
    header, seq = None, []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if header:
                parts = header.split("|")
                acc = parts[1] if len(parts) > 2 else header.split()[0]
                yield acc, header, "".join(seq)
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header:
        parts = header.split("|")
        acc = parts[1] if len(parts) > 2 else header.split()[0]
        yield acc, header, "".join(seq)
