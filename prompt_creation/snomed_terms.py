#!/usr/bin/env python3
"""
SNOMED CT Norwegian terms module.

Provides easy access to random Norwegian medical terms from SNOMED CT.
Terms are loaded from data/snomed_ct_norwegian_terms.jsonl on module import
and cached in memory.

Usage:
    from snomed_terms import get_random_snomed
    
    term = get_random_snomed()
"""

import json
import random
from pathlib import Path

# Load terms when module is imported
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
_terms_file = _project_root / "data" / "snomed_ct_norwegian_terms.jsonl"

_terms = []
try:
    with open(_terms_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'nb' in data and data['nb']:
                    nb_data = data['nb']
                    if 'akseptabel' in nb_data:
                        _terms.extend(nb_data['akseptabel'])
                    if 'tilrådd' in nb_data:
                        _terms.extend(nb_data['tilrådd'])
            except json.JSONDecodeError:
                continue
    
    # Deduplicate terms
    _terms = list(set(_terms))
except FileNotFoundError:
    raise FileNotFoundError(
        f"SNOMED CT terms file not found: {_terms_file}"
    )


def get_random_snomed():
    """
    Get a random Norwegian medical term from SNOMED CT.
    
    Returns:
        str: A random Norwegian medical term
        
    Example:
        >>> term = get_random_snomed()
        >>> print(term)
        "parathyreoidea"
    """
    return random.choice(_terms)

