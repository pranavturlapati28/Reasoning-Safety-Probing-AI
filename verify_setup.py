"""
Verify that the project setup is complete and working.
Run this before starting the main research.
"""

import sys
import os
from pathlib import Path

def check_directories():
    """Check that all required directories exist."""
    required_dirs = ['data', 'notebooks', 'src', 'results', 'logs']
    missing = []

    for dirname in required_dirs:
        if not os.path.exists(dirname):
            missing.append(dirname)

    if missing:
        print("❌ Missing directories:", ', '.join(missing))
        return False
    else:
        print("✅ All required directories present")
        return True

def check_source_files():
    """Check that all source files exist."""
    required_files = [
        'src/__init__.py',
        'src/data_generation.py',
        'src/model_inference.py',
        'src/probe_training.py',
        'src/analysis.py'
    ]
    missing = []

    for filepath in required_files:
        if not os.path.exists(filepath):
            missing.append(filepath)

    if missing:
        print("❌ Missing source files:", ', '.join(missing))
        return False
    else:
        print("✅ All source files present")
        return True

def check_imports():
    """Check that key imports work."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False

    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed")
        return False

    return True

def check_custom_modules():
    """Check that custom modules can be imported."""
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from src import data_generation, model_inference, probe_training, analysis
        print("✅ Custom modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Error importing custom modules: {e}")
        return False

def test_data_generation():
    """Test data generation module."""
    try:
        from src.data_generation import SAMPLE_QUESTIONS, generate_dataset

        dataset = generate_dataset(SAMPLE_QUESTIONS[:1], n_examples_per_type=2)

        if len(dataset) == 6:  # 3 hint types * 2 examples
            print(f"✅ Data generation works (created {len(dataset)} examples)")
            return True
        else:
            print(f"❌ Data generation unexpected output: {len(dataset)} examples")
            return False
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_probe_training():
    """Test probe training module with synthetic data."""
    try:
        import numpy as np
        from src.probe_training import LinearProbe

        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 512)
        y = np.random.randint(0, 2, 100)

        # Train probe
        probe = LinearProbe(input_dim=512)
        results = probe.train(X, y, validation_split=0.2)

        if 'val_auroc' in results:
            print(f"✅ Probe training works (val AUROC: {results['val_auroc']:.3f})")
            return True
        else:
            print("❌ Probe training missing expected outputs")
            return False
    except Exception as e:
        print(f"❌ Probe training error: {e}")
        return False

def check_gpu():
    """Check GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU available (will use CPU - slower)")

        return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    print()

    checks = [
        ("Directory structure", check_directories),
        ("Source files", check_source_files),
        ("Python packages", check_imports),
        ("Custom modules", check_custom_modules),
        ("Data generation", test_data_generation),
        ("Probe training", test_probe_training),
        ("GPU/CUDA", check_gpu),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            results.append(check_func())
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\nYou're ready to start! Run:")
        print("  jupyter notebook")
        print("  # Then open notebooks/01_exploration.ipynb")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\nPlease fix the issues above before proceeding.")
        print("If you need help, check:")
        print("  - requirements.txt")
        print("  - QUICK_START.md")
        print("  - MATS_APPLICATION_RULES.md")

    print("=" * 60)

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
