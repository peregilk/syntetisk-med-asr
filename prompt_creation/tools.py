#!/usr/bin/env python3
"""
Tools module for prompt generation.

Provides easy access to random Norwegian medical terms from SNOMED CT
and random medical scenarios from categories.

Usage:
    from tools import get_random_snomed, get_random_scenario
    
    term = get_random_snomed()
    terms = get_random_snomed(5)  # Returns list of 5 terms
    scenario = get_random_scenario()
    scenarios = get_random_scenario(3)  # Returns list of 3 scenarios
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


def get_random_snomed(N=None):
    """
    Get random Norwegian medical term(s) from SNOMED CT.
    
    Args:
        N (int, optional): Number of terms to return. If None, returns a single term.
    
    Returns:
        str or list: A random Norwegian medical term (if N is None) or a list of N terms
        
    Example:
        >>> term = get_random_snomed()
        >>> print(term)
        "parathyreoidea"
        >>> terms = get_random_snomed(5)
        >>> print(terms)
        ["parathyreoidea", "angstlidelse", ...]
    """
    if N is None:
        return random.choice(_terms)
    else:
        if N <= len(_terms):
            return random.sample(_terms, N)
        else:
            # If N > available terms, allow duplicates
            return random.choices(_terms, k=N)


def get_random_scenario(N=None):
    """
    Get random medical scenario(s) from categories.
    
    Args:
        N (int, optional): Number of scenarios to return. If None, returns a single scenario.
    
    Returns:
        str or list: A random medical scenario text (if N is None) or a list of N scenarios
        
    Example:
        >>> scenario = get_random_scenario()
        >>> print(scenario)
        "På en sykehuspoliklinikk går legen og pasienten gjennom nye funn sammen."
        >>> scenarios = get_random_scenario(3)
        >>> print(scenarios)
        ["På en sykehuspoliklinikk...", "I et rom på...", ...]
    """
    if N is None:
        return random.choice(_scenarios)
    else:
        if N <= len(_scenarios):
            return random.sample(_scenarios, N)
        else:
            # If N > available scenarios, allow duplicates
            return random.choices(_scenarios, k=N)

