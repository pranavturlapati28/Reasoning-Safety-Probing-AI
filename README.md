# Detecting Unfaithful Hint Usage in Reasoning Models

MATS 10.0 Application Project - Neel Nanda Stream

## Research Question

Can linear probes trained on reasoning model hidden states detect unfaithful hint usage—where the model changes its answer due to a hint without verbalizing that reliance—at rates significantly above chance? Is there a statistically significant difference in probe detection accuracy between reasoning models (DeepSeek-R1-Distill, QwQ) and instruction-tuned models (Qwen2.5-Instruct), as measured by AUROC and calibrated using permutation tests?

## Research Alignment

This project aligns with Neel Nanda's research interests in:
- **Chain-of-Thought Faithfulness**: Understanding when reasoning models' CoT reflects their actual reasoning
- **Reasoning Models**: Investigating how models with extended thinking behave
- **Model Biology**: Studying high-level behavioral properties
- **Applied Interpretability**: Building practical monitors for model behavior

## Project Structure

```
.
├── data/                   # Dataset storage
├── notebooks/             # Jupyter notebooks for experiments
│   ├── 01_exploration.ipynb
│   ├── 02_probe_training.ipynb
│   └── 03_analysis.ipynb
├── src/                   # Source code
│   ├── data_generation.py    # Create questions with hints
│   ├── model_inference.py    # Run models, extract hidden states
│   ├── probe_training.py     # Train and evaluate probes
│   └── analysis.py           # Qualitative analysis tools
├── results/               # Outputs, figures, tables
├── logs/                  # Experiment logs
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Start Jupyter
jupyter notebook

# 2. Open notebooks/01_exploration.ipynb

# 3. Follow the analysis workflow
```

## Time Tracking

Use Toggl or similar to track time. See MATS_APPLICATION_RULES.md for what counts toward the 20-hour limit.

---

**Remember**: The goal is to learn something interesting and communicate it clearly!
