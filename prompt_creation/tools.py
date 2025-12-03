#!/usr/bin/env python3
"""
Tools module for prompt generation.

Provides easy access to random Norwegian medical terms from SNOMED CT
and random medical scenarios from categories.

Usage:
    from tools import get_random_snomed, get_random_scenario
    
    term = get_random_snomed()
    scenario = get_random_scenario()
"""

import json
import random
from pathlib import Path

# Load terms when module is imported
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
_terms_file = _project_root / "data" / "snomed_ct_norwegian_terms.jsonl"
_scenarios_file = _project_root / "data" / "categories.jsonl"

# Load SNOMED terms
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

# Load scenarios
_scenarios = []
try:
    with open(_scenarios_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'Scenario' in data and data['Scenario']:
                    _scenarios.append(data['Scenario'])
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    raise FileNotFoundError(
        f"Scenarios file not found: {_scenarios_file}"
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


def get_random_scenario():
    """
    Get a random medical scenario from categories.
    
    Returns:
        str: A random medical scenario text
        
    Example:
        >>> scenario = get_random_scenario()
        >>> print(scenario)
        "På en sykehuspoliklinikk går legen og pasienten gjennom nye funn sammen."
    """
    return random.choice(_scenarios)

