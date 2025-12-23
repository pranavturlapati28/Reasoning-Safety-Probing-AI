"""
Relabel classification data for probe training.

Original categories:
- RESISTANT: Model ignores hint, answer unchanged
- FAITHFUL_INCORRECT: Model follows hint, verbalizes it, gets wrong answer
- UNFAITHFUL: Model follows hint, does NOT verbalize it, changes answer
- ALREADY_WRONG: Baseline was already wrong
- NO_CHANGE: Answer didn't change (merged into RESISTANT)

New binary labels for probe training:
- INFLUENCED (1): UNFAITHFUL + FAITHFUL_INCORRECT (model changed answer due to hint)
- NOT_INFLUENCED (0): RESISTANT (model ignored hint)
- Exclude ALREADY_WRONG from training (baseline was wrong, can't measure influence)
"""

import pandas as pd
import json
from pathlib import Path


def relabel_for_influence_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel data for hint influence detection.

    Returns dataframe with new 'influenced' binary label.
    """
    df = df.copy()

    # Create binary label
    def get_influence_label(category):
        if category in ['UNFAITHFUL', 'FAITHFUL_INCORRECT']:
            return 1  # Influenced
        elif category in ['RESISTANT', 'NO_CHANGE']:
            return 0  # Not influenced
        else:  # ALREADY_WRONG
            return -1  # Exclude from training

    df['influenced'] = df['category'].apply(get_influence_label)

    # Also create a more specific label for analysis
    def get_faithful_label(category):
        """Binary: was hint influence verbalized?"""
        if category == 'UNFAITHFUL':
            return 0  # Not faithful (this is what we want to detect!)
        elif category == 'FAITHFUL_INCORRECT':
            return 1  # Faithful (verbalized hint usage)
        else:
            return -1  # Not applicable

    df['faithful'] = df['category'].apply(get_faithful_label)

    return df


def analyze_label_distribution(df: pd.DataFrame, model_name: str):
    """Print statistics about label distribution."""
    print(f"\n{'='*60}")
    print(f"Label Distribution for {model_name}")
    print(f"{'='*60}")

    print("\nOriginal Categories:")
    print(df['category'].value_counts().to_string())

    # For training data (exclude ALREADY_WRONG)
    trainable = df[df['influenced'] != -1]

    print(f"\n\nTrainable Examples (excluding ALREADY_WRONG):")
    print(f"Total: {len(trainable)}")
    print(f"Influenced (UNFAITHFUL + FAITHFUL_INCORRECT): {(trainable['influenced'] == 1).sum()}")
    print(f"Not Influenced (RESISTANT): {(trainable['influenced'] == 0).sum()}")

    # Unfaithful specifically
    unfaithful_count = (df['category'] == 'UNFAITHFUL').sum()
    print(f"\n⚠️  UNFAITHFUL examples (primary target): {unfaithful_count}")

    if unfaithful_count < 10:
        print("   WARNING: Very few unfaithful examples - consider 'influence' detection instead")

    print(f"\nExcluded (ALREADY_WRONG): {(df['influenced'] == -1).sum()}")


def merge_all_models(
    qwen05b_path: str,
    qwen7b_path: str,
    deepseek_path: str
) -> pd.DataFrame:
    """
    Merge all model results into single dataframe for analysis.
    """
    qwen05b = pd.read_csv(qwen05b_path)
    qwen7b = pd.read_csv(qwen7b_path)
    deepseek = pd.read_csv(deepseek_path)

    # Add model identifier
    qwen05b['model'] = 'Qwen-0.5B-Instruct'
    qwen05b['model_type'] = 'instruct'
    qwen05b['model_size'] = '0.5B'

    qwen7b['model'] = 'Qwen-7B-Instruct'
    qwen7b['model_type'] = 'instruct'
    qwen7b['model_size'] = '7B'

    deepseek['model'] = 'DeepSeek-R1-Distill-7B'
    deepseek['model_type'] = 'reasoning'
    deepseek['model_size'] = '7B'

    # Relabel all
    qwen05b = relabel_for_influence_detection(qwen05b)
    qwen7b = relabel_for_influence_detection(qwen7b)
    deepseek = relabel_for_influence_detection(deepseek)

    # Analyze each
    analyze_label_distribution(qwen05b, 'Qwen-0.5B-Instruct')
    analyze_label_distribution(qwen7b, 'Qwen-7B-Instruct')
    analyze_label_distribution(deepseek, 'DeepSeek-R1-Distill-7B')

    # Combine
    combined = pd.concat([qwen05b, qwen7b, deepseek], ignore_index=True)

    return combined


if __name__ == "__main__":
    # Paths to classified results
    data_dir = Path("/Users/pranavturlapati/Reasoning-Safety-Probing-AI/notebooks")

    qwen05b_path = data_dir / "phase1_48questions_classified.csv"
    qwen7b_path = data_dir / "qwen7b_48questions_classified.csv"
    deepseek_path = data_dir / "deepseek_48questions_classified.csv"

    # Check which files exist
    print("Checking for classified data files...")
    for name, path in [
        ("Qwen 0.5B", qwen05b_path),
        ("Qwen 7B", qwen7b_path),
        ("DeepSeek R1", deepseek_path)
    ]:
        if path.exists():
            print(f"✓ Found {name}: {path}")
        else:
            print(f"✗ Missing {name}: {path}")

    # Merge all models
    combined = merge_all_models(str(qwen05b_path), str(qwen7b_path), str(deepseek_path))

    # Save merged dataset
    output_path = data_dir / "all_models_relabeled.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved merged dataset to: {output_path}")

    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS (All Models Combined)")
    print(f"{'='*60}")

    trainable = combined[combined['influenced'] != -1]
    print(f"\nTotal trainable examples: {len(trainable)}")
    print(f"  - Influenced: {(trainable['influenced'] == 1).sum()}")
    print(f"  - Not Influenced: {(trainable['influenced'] == 0).sum()}")

    # By model type
    print("\nBy Model Type:")
    for model_type in ['instruct', 'reasoning']:
        subset = trainable[trainable['model_type'] == model_type]
        influenced = (subset['influenced'] == 1).sum()
        total = len(subset)
        print(f"  {model_type.upper()}: {influenced}/{total} influenced ({100*influenced/total:.1f}%)")

    # Unfaithful breakdown
    print("\nUnfaithful Examples by Model:")
    for model in combined['model'].unique():
        subset = combined[combined['model'] == model]
        unfaithful = (subset['category'] == 'UNFAITHFUL').sum()
        print(f"  {model}: {unfaithful}")

    print("\n✓ Data relabeling complete!")
    print(f"  Next step: Extract hidden states for probe training")
