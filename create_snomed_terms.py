#!/usr/bin/env python3
"""
Extract Norwegian medical terms from SNOMED CT JSONL file.

This script reads data/snomed_ct_norwegian_terms.jsonl, extracts terms from
the nb.akseptabel and nb.tilrådd fields, deduplicates them, and saves to an
optimized JSONL format in data/ for fast loading.

Usage:
    python create_snomed_terms.py
"""

import json
import time
from pathlib import Path


def main():
    """Main function to extract and process SNOMED CT terms."""
    input_file = Path("data/snomed_ct_norwegian_terms.jsonl")
    output_file = Path("data/fagterm.jsonl")
    
    # Read and parse original file
    print("Reading original JSONL file...")
    start_time = time.perf_counter()
    
    fagterm = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'nb' in data and data['nb']:
                    nb_data = data['nb']
                    if 'akseptabel' in nb_data:
                        fagterm.extend(nb_data['akseptabel'])
                    if 'tilrådd' in nb_data:
                        fagterm.extend(nb_data['tilrådd'])
            except json.JSONDecodeError:
                continue
    
    original_time = time.perf_counter() - start_time
    
    # Deduplicate
    unique_terms = list(set(fagterm))
    num_unique = len(unique_terms)
    
    print(f"\nOriginal file processing:")
    print(f"  Time: {original_time:.4f} seconds")
    print(f"  Total terms extracted: {len(fagterm)}")
    print(f"  Unique terms: {num_unique}")
    
    # Save to optimized JSONL
    print(f"\nSaving deduplicated terms to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in unique_terms:
            f.write(json.dumps(term, ensure_ascii=False) + '\n')
    
    # Read back optimized file and measure performance
    print(f"\nReading optimized JSONL file...")
    start_time = time.perf_counter()
    
    loaded_terms = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                term = json.loads(line.strip())
                loaded_terms.append(term)
            except json.JSONDecodeError:
                continue
    
    optimized_time = time.perf_counter() - start_time
    
    print(f"\nOptimized file processing:")
    print(f"  Time: {optimized_time:.4f} seconds")
    print(f"  Terms loaded: {len(loaded_terms)}")
    
    # Performance comparison
    speedup = original_time / optimized_time if optimized_time > 0 else 0
    print(f"\nPerformance comparison:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {original_time - optimized_time:.4f} seconds")
    
    # Store in fagterm for later use
    fagterm = loaded_terms
    print(f"\nTerms stored in 'fagterm' variable ({len(fagterm)} unique terms)")


if __name__ == "__main__":
    main()

