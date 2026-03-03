"""
Consistency Test: Run self-compile N times and compare outputs.

Tests whether Motherlabs produces structurally stable blueprints
across multiple runs with the same input.
"""
import sys
import json
import hashlib
from typing import List, Dict, Any

sys.path.insert(0, '.')

from core.engine import MotherlabsEngine
from core.schema import check_canonical_coverage, check_canonical_relationships


def normalize_blueprint(bp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize blueprint for comparison (sort lists, remove timestamps)."""
    normalized = {}

    # Sort components by name
    if 'components' in bp:
        normalized['components'] = sorted(
            bp['components'],
            key=lambda c: c.get('name', '')
        )

    # Sort relationships by from+to+type
    if 'relationships' in bp:
        normalized['relationships'] = sorted(
            bp['relationships'],
            key=lambda r: f"{r.get('from', '')}-{r.get('to', '')}-{r.get('type', '')}"
        )

    # Sort constraints by description
    if 'constraints' in bp:
        normalized['constraints'] = sorted(
            bp['constraints'],
            key=lambda c: c.get('description', '')
        )

    # Keep unresolved as-is (order may matter)
    if 'unresolved' in bp:
        normalized['unresolved'] = sorted(bp.get('unresolved', []))

    return normalized


def structural_hash(bp: Dict[str, Any]) -> str:
    """Create hash of structural elements only (names, types, relationships)."""
    structure = {
        'component_names': sorted([c.get('name') for c in bp.get('components', [])]),
        'component_types': sorted([f"{c.get('name')}:{c.get('type')}" for c in bp.get('components', [])]),
        'relationships': sorted([
            f"{r.get('from')}->{r.get('to')}:{r.get('type')}"
            for r in bp.get('relationships', [])
        ])
    }
    return hashlib.md5(json.dumps(structure, sort_keys=True).encode()).hexdigest()[:12]


def compare_blueprints(bp1: Dict, bp2: Dict) -> Dict[str, Any]:
    """Compare two blueprints and return differences."""
    diff = {
        'component_count_diff': len(bp1.get('components', [])) - len(bp2.get('components', [])),
        'relationship_count_diff': len(bp1.get('relationships', [])) - len(bp2.get('relationships', [])),
        'missing_in_2': [],
        'missing_in_1': [],
        'structural_match': False
    }

    names1 = {c.get('name') for c in bp1.get('components', [])}
    names2 = {c.get('name') for c in bp2.get('components', [])}

    diff['missing_in_2'] = list(names1 - names2)
    diff['missing_in_1'] = list(names2 - names1)
    diff['structural_match'] = structural_hash(bp1) == structural_hash(bp2)

    return diff


def run_consistency_test(provider: str, num_runs: int = 3):
    """Run self-compile multiple times and compare."""
    print(f"\n{'='*60}")
    print(f"CONSISTENCY TEST: {provider.upper()}")
    print(f"Running self-compile {num_runs} times")
    print('='*60)

    results = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        engine = MotherlabsEngine(
            provider=provider,
            on_insight=lambda x: None  # Suppress
        )

        result = engine.self_compile()

        # Check for valid blueprint rather than strict success
        # (success=False may just mean verify found issues, but blueprint exists)
        bp = result.blueprint
        if not bp or not bp.get('components'):
            print(f"  FAILED: No blueprint produced. Error: {result.error}")
            results.append(None)
            continue
        comp_cov = check_canonical_coverage(bp)
        rel_cov = check_canonical_relationships(bp)

        run_data = {
            'blueprint': bp,
            'normalized': normalize_blueprint(bp),
            'structural_hash': structural_hash(bp),
            'component_count': len(bp.get('components', [])),
            'relationship_count': len(bp.get('relationships', [])),
            'component_coverage': comp_cov['coverage'],
            'relationship_coverage': rel_cov['coverage'],
            'component_names': sorted([c.get('name') for c in bp.get('components', [])])
        }

        results.append(run_data)

        print(f"  Components: {run_data['component_count']}")
        print(f"  Relationships: {run_data['relationship_count']}")
        print(f"  Comp coverage: {run_data['component_coverage']:.0%}")
        print(f"  Rel coverage: {run_data['relationship_coverage']:.0%}")
        print(f"  Structural hash: {run_data['structural_hash']}")

    # Analyze consistency
    print(f"\n{'='*60}")
    print("CONSISTENCY ANALYSIS")
    print('='*60)

    valid_results = [r for r in results if r is not None]

    if len(valid_results) < 2:
        print("ERROR: Not enough successful runs to compare")
        return {'success': False, 'error': 'insufficient_runs'}

    # Check structural hashes
    hashes = [r['structural_hash'] for r in valid_results]
    unique_hashes = set(hashes)

    print(f"\nStructural hashes: {hashes}")
    print(f"Unique structures: {len(unique_hashes)}")

    if len(unique_hashes) == 1:
        print("\n✅ PERFECT CONSISTENCY: All runs produced identical structure")
        consistency_score = 1.0
    else:
        print(f"\n⚠️  VARIATION DETECTED: {len(unique_hashes)} different structures")
        consistency_score = 1.0 / len(unique_hashes)

    # Detailed comparison
    print("\n--- Pairwise Comparison ---")
    for i in range(len(valid_results)):
        for j in range(i+1, len(valid_results)):
            diff = compare_blueprints(
                valid_results[i]['blueprint'],
                valid_results[j]['blueprint']
            )
            print(f"\nRun {i+1} vs Run {j+1}:")
            print(f"  Structural match: {diff['structural_match']}")
            print(f"  Component diff: {diff['component_count_diff']}")
            print(f"  Relationship diff: {diff['relationship_count_diff']}")
            if diff['missing_in_2']:
                print(f"  Missing in run {j+1}: {diff['missing_in_2']}")
            if diff['missing_in_1']:
                print(f"  Missing in run {i+1}: {diff['missing_in_1']}")

    # Component name consistency
    print("\n--- Component Names Across Runs ---")
    all_names = set()
    for r in valid_results:
        all_names.update(r['component_names'])

    print(f"Union of all components: {len(all_names)}")
    for name in sorted(all_names):
        presence = ['✓' if name in r['component_names'] else '✗' for r in valid_results]
        print(f"  {name}: {' '.join(presence)}")

    # Summary
    summary = {
        'provider': provider,
        'num_runs': num_runs,
        'successful_runs': len(valid_results),
        'unique_structures': len(unique_hashes),
        'consistency_score': consistency_score,
        'all_hashes': hashes,
        'perfect_consistency': len(unique_hashes) == 1
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Provider: {provider}")
    print(f"Runs: {len(valid_results)}/{num_runs} successful")
    print(f"Consistency score: {consistency_score:.0%}")
    print(f"Perfect consistency: {summary['perfect_consistency']}")

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Consistency test for Motherlabs')
    parser.add_argument('--provider', default='grok', help='Provider to test')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    result = run_consistency_test(args.provider, args.runs)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
