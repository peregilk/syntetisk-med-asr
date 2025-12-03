#!/usr/bin/env python3
"""
Generate prompts from template file.

Reads a template file and generates prompts using random scenarios and SNOMED terms.
Outputs prompts in JSONL format.

Usage:
    python create_prompts.py --limit 100 --output_file prompts.jsonl
"""

import argparse
import glob
import json
import string
from pathlib import Path

from tools import get_random_scenario, get_random_snomed


def main():
    """Main function to generate prompts."""
    parser = argparse.ArgumentParser(
        description="Generate prompts from template file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of prompts to generate (required)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file name (will be saved in prompts/ folder, e.g., prompts.jsonl)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="a.txt",
        help="Template file name or glob pattern (default: a.txt, looks in templates/ folder, e.g., *.txt or template/*)"
    )
    
    args = parser.parse_args()
    
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    templates_dir = project_root / "templates"
    output_file = project_root / "prompts" / args.output_file
    
    # Check if template contains wildcards
    if '*' in args.template or '?' in args.template:
        # Use glob to find matching templates
        template_pattern = str(templates_dir / args.template)
        template_files = sorted(glob.glob(template_pattern))
        if not template_files:
            raise FileNotFoundError(f"No templates found matching pattern: {args.template}")
        print(f"Found {len(template_files)} template(s) matching pattern: {args.template}")
    else:
        # Single template file
        template_file = templates_dir / args.template
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        template_files = [str(template_file)]
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all templates
    templates_data = []
    for template_path in template_files:
        template_file = Path(template_path)
        template_name = template_file.stem
        template_filename = template_file.name
        
        # Read template
        with open(template_file, 'r', encoding='utf-8') as tf:
            template_content = tf.read()
        
        templates_data.append({
            'name': template_name,
            'filename': template_filename,
            'template': string.Template(template_content)
        })
    
    num_templates = len(templates_data)
    prompts_per_template = args.limit // num_templates
    
    print(f"Found {num_templates} template(s). Generating {args.limit} prompts total ({prompts_per_template} per template)...")
    
    # Generate prompts, cycling through templates
    template_counters = {t['name']: 0 for t in templates_data}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(args.limit):
            # Cycle through templates
            template_idx = i % num_templates
            template_data = templates_data[template_idx]
            
            # Increment counter for this template
            template_counters[template_data['name']] += 1
            counter = template_counters[template_data['name']]
            
            # Get random data (scenarios and SNOMED terms are still random)
            scenario = get_random_scenario()
            snomed_required = get_random_snomed(1)[0]  # Get first element from list
            snomed_optional_list = get_random_snomed(10)
            
            # Format optional terms as comma-separated list
            snomed_optional = ", ".join(snomed_optional_list)
            
            # Substitute template variables
            prompt_text = template_data['template'].substitute(
                scenario=scenario,
                snomed_required=snomed_required,
                snomed_optional=snomed_optional
            )
            
            # Write as JSON line with unique ID (includes template name)
            prompt_obj = {
                "id": f"{template_data['name']}_{counter}",
                "template": template_data['filename'],
                "prompt": prompt_text
            }
            f.write(json.dumps(prompt_obj, ensure_ascii=False) + '\n')
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{args.limit} prompts...")
    
    print(f"Done! Generated {args.limit} prompts ({prompts_per_template} per template × {num_templates} template(s)) to {output_file}")


if __name__ == "__main__":
    main()

