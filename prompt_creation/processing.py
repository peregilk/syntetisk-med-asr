#!/usr/bin/env python3
"""Preprocessing utilities for SNOMED and related artifacts."""

import argparse
import json
import math
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

# Split on special characters (except hyphen) for enkeltord/terms.
_TERM_SPLIT_PATTERN = re.compile(r"[^\w-]+", re.UNICODE)
_SPACE_SPLIT_PATTERN = re.compile(r"\s+")
_CHEMICAL_SPLIT_PATTERN = re.compile(r"[^0-9A-Za-zÆØÅæøå]+")

# Remove phrases with special characters (except hyphen) for TTS-ready phrases.
_TERM_ALLOWED_PATTERN = re.compile(r"^[0-9A-Za-zÆØÅæøå-]+$")  # Terms allow only letters/digits and hyphen (no spaces/commas).
_PHRASE_DISALLOWED_PATTERN = re.compile(r"[^0-9A-Za-zÆØÅæøå ,\-−]")  # Phrases may include spaces/commas/hyphen/minus.
_HYPHENATED_NUMBER_PATTERN = re.compile(r"^\d+(?:-\d+)+$")


def normalize_identifier(term: str) -> str:
    """Normalize a term for ID generation (ASCII-ish, lowercase, no punctuation)."""
    cleaned = term.lower().strip()
    cleaned = unicodedata.normalize("NFC", cleaned)
    return re.sub(r"[^a-zæøå]", "", cleaned)


def _base36_encode(value: int, width: int) -> str:
    """Encode a non-negative integer as zero-padded base36."""
    if value < 0:
        raise ValueError("Base36 encoding requires non-negative integers")
    if value == 0:
        encoded = "0"
    else:
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        parts: list[str] = []
        current = value
        while current:
            current, remainder = divmod(current, 36)
            parts.append(digits[remainder])
        encoded = "".join(reversed(parts))
    return encoded.zfill(width)


def build_term_id_map(entries: list[dict], width: int = 3) -> dict[str, str]:
    """Build stable term IDs based on normalized lexicographic order."""
    sortable: list[tuple[str, str]] = []
    for entry in entries:
        term = entry.get("term")
        if not isinstance(term, str) or not term.strip():
            continue
        normalized = normalize_identifier(term)
        sortable.append((normalized, term))

    sortable.sort(key=lambda item: (item[0], item[1]))

    term_ids: dict[str, str] = {}
    for index, (normalized, term) in enumerate(sortable):
        prefix = normalized[:6] if normalized else "term"
        term_ids[term] = f"{prefix}_{_base36_encode(index, width)}"
    return term_ids


def load_snomed_terms(path: Path) -> list[str]:
    """Load Norwegian SNOMED phrases and terms from JSONL."""
    terms: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"SNOMED CT terms file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        nb_data = data.get("nb")
        if not isinstance(nb_data, dict):
            continue
        for key in ["akseptabel", "tilrådd"]:
            values = nb_data.get(key)
            if isinstance(values, list):
                terms.extend([v for v in values if isinstance(v, str) and v.strip()])
    return terms


def extract_phrase_candidates(phrase: str) -> list[str]:
    """Extract candidate phrases for reuse in TTS.

    Pipeline:
    1) Pull out parenthetical content as separate expressions.
    2) Split on colon and treat each part as its own expression.
    3) Split on " - " (space-hyphen-space) and treat each part as its own expression.
    4) Keep remaining expressions as-is.
    """
    candidates: list[str] = []
    if not phrase:
        return candidates

    parenthetical_parts = re.findall(r"\(([^)]+)\)", phrase)
    candidates.extend([part.strip() for part in parenthetical_parts if part.strip()])

    colon_parts = [part.strip() for part in phrase.split(":") if part.strip()]
    for part in colon_parts:
        dash_parts = [p.strip() for p in part.split(" - ") if p.strip()]
        candidates.extend(dash_parts)

    return candidates


def clean_phrase(phrase: str) -> str | None:
    """Normalize a phrase to a TTS-friendly form or drop it."""
    trimmed = phrase.strip()
    if not trimmed:
        return None

    trimmed = trimmed.replace("%", "prosent")
    if _PHRASE_DISALLOWED_PATTERN.search(trimmed):
        return None

    return trimmed


def build_cleaned_phrases(snomed_terms: list[str]) -> list[str]:
    """Build a deduplicated list of cleaned phrases for TTS."""
    cleaned_phrases: list[str] = []
    for phrase in snomed_terms:
        for candidate in extract_phrase_candidates(phrase):
            cleaned = clean_phrase(candidate)
            if cleaned is not None:
                cleaned_phrases.append(cleaned)
    return list(dict.fromkeys(cleaned_phrases))


def save_phrases(phrases: list[str], output_path: Path) -> None:
    """Write cleaned phrases to JSONL."""
    with output_path.open("w", encoding="utf-8") as handle:
        for phrase in phrases:
            handle.write(json.dumps({"phrase": phrase}, ensure_ascii=False) + "\n")


def _normalize_term(term: str, stemmer: SnowballStemmer) -> str:
    """Normalize term for variant merging."""
    parts = term.split("-")
    normalized_parts = []
    for part in parts:
        if not part:
            normalized_parts.append(part)
            continue
        lower = part.lower()
        if lower.isnumeric():
            normalized_parts.append(lower)
            continue
        normalized_parts.append(stemmer.stem(lower))
    return "-".join(normalized_parts)


def _choose_representative_term(
    variants: set[str],
    snomed_variant_counts: dict[str, int],
    corpus_counts: dict[str, int],
) -> str:
    """Select a representative spelling from variants for output."""
    return sorted(
        variants,
        key=lambda term: (
            -snomed_variant_counts.get(term, 0),
            -corpus_counts.get(term, 0),
            len(term),
            term,
        ),
    )[0]


def _decode_bytes(content: bytes) -> str:
    """Decode bytes with UTF-8 and fall back to latin-1."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="replace")


def _read_jsonl_lines(path: Path) -> list[dict]:
    """Read JSONL from plain or zipped files, skipping invalid lines."""
    if not path.exists():
        return []
    if path.suffix == ".zip":
        items: list[dict] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                with archive.open(name) as handle:
                    content = _decode_bytes(handle.read())
                for raw in content.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return items

    items = []
    content = _decode_bytes(path.read_bytes())
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def load_occurrence_entries(paths: list[Path]) -> list[dict]:
    """Load and deduplicate corpus occurrence entries."""
    entries: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for item in _read_jsonl_lines(path):
            word = item.get("word")
            count = item.get("count")
            if isinstance(word, str) and word and isinstance(count, int):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(item)
    return entries


def build_corpus_counts(occurrence_entries: list[dict]) -> dict[str, int]:
    """Aggregate corpus counts into a lookup dictionary."""
    corpus_counts: dict[str, int] = defaultdict(int)
    for entry in occurrence_entries:
        corpus_counts[entry["word"]] += entry["count"]
    return dict(corpus_counts)


def save_occurrences(occurrence_entries: list[dict], output_path: Path) -> None:
    """Write combined corpus occurrences to JSONL."""
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in sorted(occurrence_entries, key=lambda item: item["count"], reverse=True):
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _is_complex_chemical(term: str) -> bool:
    """Return True when a token is likely complex chemical nomenclature."""
    if re.fullmatch(r"\d+[\-,]\d+", term):
        return False

    prefix_pattern = r"^(cis|trans|ortho|para|meta|[NOLSD])\-[a-z0-9]"
    chem_suffixes = r"(ase|yl|id|at|in|an|ol|en)\b"
    structure_pattern = r"(\d+,\d+|alfa|beta|gamma|delta|epsilon|kappa)"
    bond_pattern = r"[A-Za-zÆØÅæøå]-\d+(,\d+)*-[A-Za-zÆØÅæøå]"

    if re.search(prefix_pattern, term):
        return True

    if re.search(bond_pattern, term, re.IGNORECASE):
        return True

    if re.search(chem_suffixes, term, re.IGNORECASE) and re.search(
        structure_pattern,
        term,
        re.IGNORECASE,
    ):
        return True

    if len(term) > 12 and term.count("-") >= 2 and re.search(r"\d", term):
        return True

    return False


def tokenize_terms(text: str) -> list[str]:
    """Split a phrase into enkelttokens while preserving hyphens."""
    space_tokens = [t for t in _SPACE_SPLIT_PATTERN.split(text) if t]
    pre_tokens: list[str] = []
    for token in space_tokens:
        if _is_complex_chemical(token):
            pre_tokens.extend([t for t in _CHEMICAL_SPLIT_PATTERN.split(token) if t])
        else:
            pre_tokens.append(token)

    tokens: list[str] = []
    for token in pre_tokens:
        for part in _TERM_SPLIT_PATTERN.split(token):
            if not part:
                continue
            stripped = part.strip("-")
            if not stripped:
                continue
            if stripped.isdigit():
                continue
            if _HYPHENATED_NUMBER_PATTERN.match(stripped):
                continue
            if _TERM_ALLOWED_PATTERN.match(stripped):
                tokens.append(stripped)
    return tokens


def build_term_set(snomed_terms: list[str]) -> set[str]:
    """Build a set of SNOMED terms for usage counting."""
    term_set: set[str] = set()
    for phrase in snomed_terms:
        for token in tokenize_terms(phrase):
            if len(token) > 1:
                term_set.add(token)
    return term_set


def build_variant_maps(
    term_set: set[str],
    corpus_counts: dict[str, int],
    merge_enabled: bool,
) -> tuple[SnowballStemmer | None, dict[str, set[str]], set[str], dict[str, int]]:
    """Prepare variant maps and merged corpus counts if enabled."""
    if not merge_enabled:
        return None, {}, set(), corpus_counts

    stemmer = SnowballStemmer("norwegian")
    variant_map: dict[str, set[str]] = defaultdict(set)
    for term in term_set:
        normalized = _normalize_term(term, stemmer)
        variant_map[normalized].add(term)

    normalized_term_set = set(variant_map.keys())
    merged_corpus_counts: dict[str, int] = defaultdict(int)
    for term, count in corpus_counts.items():
        merged_corpus_counts[_normalize_term(term, stemmer)] += count

    return stemmer, variant_map, normalized_term_set, dict(merged_corpus_counts)


def count_term_usage(
    cleaned_phrases: list[str],
    term_set: set[str],
    merge_enabled: bool,
    stemmer: SnowballStemmer | None,
) -> tuple[dict[str, int], dict[str, set[str]], dict[str, int]]:
    """Count term occurrences inside cleaned phrases."""
    snomed_counts: dict[str, int] = defaultdict(int)
    usage_map: dict[str, set[str]] = defaultdict(set)
    variant_snomed_counts: dict[str, int] = defaultdict(int)

    for phrase in cleaned_phrases:
        tokens = tokenize_terms(phrase)
        for token in tokens:
            if token not in term_set:
                continue
            variant_snomed_counts[token] += 1
            if merge_enabled:
                normalized = _normalize_term(token, stemmer)
                snomed_counts[normalized] += 1
                usage_map[normalized].add(phrase)
            else:
                snomed_counts[token] += 1
                usage_map[token].add(phrase)

    return snomed_counts, usage_map, variant_snomed_counts


def build_entries(
    term_set: set[str],
    normalized_term_set: set[str],
    variant_map: dict[str, set[str]],
    snomed_counts: dict[str, int],
    corpus_counts: dict[str, int],
    usage_map: dict[str, set[str]],
    variant_snomed_counts: dict[str, int],
    merge_enabled: bool,
) -> list[dict]:
    """Build SNOMED entries with usage information."""
    entries: list[dict] = []
    if merge_enabled:
        for normalized in sorted(normalized_term_set):
            variants = variant_map.get(normalized, set())
            if not variants:
                continue
            representative = _choose_representative_term(
                variants,
                variant_snomed_counts,
                corpus_counts,
            )
            entries.append(
                {
                    "term": representative,
                    "snomed_count": snomed_counts.get(normalized, 0),
                    "corpus_count": corpus_counts.get(normalized, 0),
                    "usage": sorted(usage_map.get(normalized, set())),
                    "variants": sorted(variants),
                }
            )
    else:
        for term in sorted(term_set):
            entries.append(
                {
                    "term": term,
                    "snomed_count": snomed_counts.get(term, 0),
                    "corpus_count": corpus_counts.get(term, 0),
                    "usage": sorted(usage_map.get(term, set())),
                }
            )
    return entries


def save_terms(entries: list[dict], ranks: list[int], output_path: Path) -> None:
    """Save terms with counts and rank to JSONL in the provided order."""
    with output_path.open("w", encoding="utf-8") as handle:
        for entry, rank in zip(entries, ranks):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "term": entry["term"],
                        "snomed_count": entry["snomed_count"],
                        "corpus_count": entry["corpus_count"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_unfiltered(entries: list[dict], output_path: Path) -> None:
    """Save unfiltered SNOMED entries to JSONL."""
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def filter_entries(
    entries: list[dict],
    min_snomed_count: int,
    max_corpus_count: int,
) -> list[dict]:
    """Filter entries to keep common-in-SNOMED but rare-in-corpus terms."""
    return [
        entry
        for entry in entries
        if entry["snomed_count"] >= min_snomed_count
        and entry["corpus_count"] <= max_corpus_count
    ]


def score_entries(
    entries: list[dict],
    rank_alpha: float,
) -> tuple[list[dict], list[int]]:
    """Rank entries by number of terms with equal-or-better log score."""
    def _rank_score(item: dict) -> float:
        return (
            rank_alpha * math.log1p(item.get("snomed_count", 0))
            + (1.0 - rank_alpha) * math.log1p(item.get("corpus_count", 0))
        )

    ranked_entries = sorted(entries, key=_rank_score, reverse=True)
    raw_scores = [_rank_score(entry) for entry in ranked_entries]
    n_entries = len(ranked_entries)
    if n_entries == 0:
        return ranked_entries, []

    scores: list[int] = [0] * n_entries
    index = 0
    while index < n_entries:
        group_score = raw_scores[index]
        group_end = index
        while group_end < n_entries and raw_scores[group_end] == group_score:
            group_end += 1
        score_value = group_end
        for position in range(index, group_end):
            scores[position] = score_value
        index = group_end

    return ranked_entries, scores


def save_snomed(
    ranked_entries: list[dict],
    ranks: list[int],
    output_path: Path,
    merge_enabled: bool,
    term_id_map: dict[str, str],
) -> None:
    """Save final SNOMED entries with rank and term ID to JSONL."""
    with output_path.open("w", encoding="utf-8") as handle:
        for entry, rank in zip(ranked_entries, ranks):
            term_id = term_id_map.get(entry["term"], "term_000")
            output_entry = {
                "rank": rank,
                "term_id": term_id,
                "term": entry["term"],
                "snomed_count": entry["snomed_count"],
                "corpus_count": entry["corpus_count"],
            }
            if merge_enabled:
                variants = entry.get("variants", [])
                output_entry["n_variants"] = len(variants)
                output_entry["variants"] = variants
            output_entry["usage"] = entry.get("usage", [])
            handle.write(json.dumps(output_entry, ensure_ascii=False) + "\n")


def get_occurrence_paths(word_occurrence_dir: Path) -> list[Path]:
    """Resolve occurrence sources including zip fallbacks."""
    occurrence_sources = [
        word_occurrence_dir / "common.jsonl",
        word_occurrence_dir / "uncommon.jsonl",
        word_occurrence_dir / "rare.jsonl",
        word_occurrence_dir / "very_rare.jsonl",
        word_occurrence_dir / "extremely_rare.jsonl",
        word_occurrence_dir / "ultra_rare.jsonl.zip",
        word_occurrence_dir / "hapax_doubleton.jsonl.zip",
    ]
    occurrence_paths: list[Path] = []
    for path in occurrence_sources:
        if path.exists():
            occurrence_paths.append(path)
        elif path.suffix == ".zip":
            fallback = path.with_suffix("")
            if fallback.exists():
                occurrence_paths.append(fallback)
    return occurrence_paths


def preprocess_snomed(
    terms_file: Path,
    preprocessed_dir: Path,
    word_occurrence_dir: Path,
    min_snomed_count: int = 0,
    max_corpus_count: int = 100,
    keep_unfiltered: bool = False,
    merge_variants: bool = False,
    rank_alpha: float = 0.7,
    rank_cap: int = 10000,
) -> dict[str, Path]:
    """Preprocess SNOMED phrases and terms into reusable JSONL artifacts."""
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    phrases_file = preprocessed_dir / "snomed_phrases.jsonl"
    terms_file_out = preprocessed_dir / "snomed_terms.jsonl"
    occurrences_file = preprocessed_dir / "occurences.jsonl"
    snomed_file = preprocessed_dir / "snomed.jsonl"

    snomed_terms = load_snomed_terms(terms_file)
    cleaned_phrases = build_cleaned_phrases(snomed_terms)
    save_phrases(cleaned_phrases, phrases_file)

    occurrence_paths = get_occurrence_paths(word_occurrence_dir)
    occurrence_entries = load_occurrence_entries(occurrence_paths)
    corpus_counts = build_corpus_counts(occurrence_entries)
    save_occurrences(occurrence_entries, occurrences_file)

    term_set = build_term_set(snomed_terms)
    stemmer, variant_map, normalized_term_set, corpus_counts = build_variant_maps(
        term_set,
        corpus_counts,
        merge_variants,
    )

    snomed_counts, usage_map, variant_snomed_counts = count_term_usage(
        cleaned_phrases,
        term_set,
        merge_variants,
        stemmer,
    )

    entries = build_entries(
        term_set,
        normalized_term_set,
        variant_map,
        snomed_counts,
        corpus_counts,
        usage_map,
        variant_snomed_counts,
        merge_variants,
    )
    ranked_all_entries, all_ranks = score_entries(entries, rank_alpha)
    save_terms(ranked_all_entries, all_ranks, terms_file_out)

    if keep_unfiltered:
        save_unfiltered(entries, preprocessed_dir / "snomed_unfiltered.jsonl")

    filtered_entries = filter_entries(entries, min_snomed_count, max_corpus_count)
    ranked_entries, ranks = score_entries(filtered_entries, rank_alpha)
    if rank_cap is not None:
        capped = [
            (entry, rank)
            for entry, rank in zip(ranked_entries, ranks)
            if rank <= rank_cap
        ]
        ranked_entries = [entry for entry, _rank in capped]
        ranks = [rank for _entry, rank in capped]
    term_id_map = build_term_id_map(ranked_entries)
    save_snomed(ranked_entries, ranks, snomed_file, merge_variants, term_id_map)

    return {
        "phrases": phrases_file,
        "terms": terms_file_out,
        "occurrences": occurrences_file,
        "snomed": snomed_file,
    }


def _resolve_default_paths() -> tuple[Path, Path, Path]:
    """Resolve default file locations relative to the project root."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    terms_file = project_root / "data" / "snomed_ct_norwegian_terms.jsonl"
    preprocessed_dir = project_root / "data" / "preprocessed"
    word_occurrence_dir = project_root / "data" / "word_occurences"
    return terms_file, preprocessed_dir, word_occurrence_dir


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for SNOMED preprocessing."""
    terms_file, preprocessed_dir, word_occurrence_dir = _resolve_default_paths()
    parser = argparse.ArgumentParser(
        description="Preprocess SNOMED phrases and terms into JSONL artifacts.",
    )
    parser.add_argument(
        "--terms-file",
        type=Path,
        default=terms_file,
        help="Path to snomed_ct_norwegian_terms.jsonl",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=preprocessed_dir,
        help="Output directory for preprocessed JSONL artifacts",
    )
    parser.add_argument(
        "--word-occurrence-dir",
        type=Path,
        default=word_occurrence_dir,
        help="Directory holding word occurrence JSONL/ZIP files",
    )
    parser.add_argument(
        "--min-snomed-count",
        type=int,
        default=0,
        help="Minimum SNOMED usage count (default from tools.py)",
    )
    parser.add_argument(
        "--max-corpus-count",
        type=int,
        default=100,
        help="Maximum corpus count (default from tools.py)",
    )
    parser.add_argument(
        "--keep-unfiltered",
        action="store_true",
        default=False,
        help="Also save unfiltered SNOMED entries",
    )
    parser.add_argument(
        "--no-merge-variants",
        dest="merge_variants",
        action="store_false",
        default=True,
        help="Disable merging of term variants",
    )
    parser.add_argument(
        "--rank-alpha",
        type=float,
        default=0.7,
        help="Weight for SNOMED vs corpus counts (default from tools.py)",
    )
    parser.add_argument(
        "--rank-cap",
        type=int,
        default=10000,
        help="Maximum rank to keep (default from tools.py)",
    )
    return parser


def main() -> None:
    """Run SNOMED preprocessing from CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    outputs = preprocess_snomed(
        terms_file=args.terms_file,
        preprocessed_dir=args.preprocessed_dir,
        word_occurrence_dir=args.word_occurrence_dir,
        min_snomed_count=args.min_snomed_count,
        max_corpus_count=args.max_corpus_count,
        keep_unfiltered=args.keep_unfiltered,
        merge_variants=args.merge_variants,
        rank_alpha=args.rank_alpha,
        rank_cap=args.rank_cap,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
