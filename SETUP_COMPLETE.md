# ✅ Setup Complete!

## Project Structure

```
Reasoning-Safety-Probing-AI/
├── 📄 MATS_APPLICATION_RULES.md    # Complete MATS application guidelines
├── 📄 README.md                     # Project overview
├── 📄 RESEARCH_PLAN.md              # Detailed 20-hour plan
├── 📄 QUICK_START.md                # Quick reference guide
├── 📄 verify_setup.py               # Setup verification script
│
├── 📁 src/                          # Source code
│   ├── __init__.py
│   ├── data_generation.py           # Create questions with hints
│   ├── model_inference.py           # Run models, extract hidden states
│   ├── probe_training.py            # Train linear probes, statistical tests
│   └── analysis.py                  # Qualitative analysis tools
│
├── 📁 notebooks/                    # Jupyter notebooks
│   └── 01_exploration.ipynb         # Start here!
│
├── 📁 data/                         # Dataset storage (empty)
├── 📁 results/                      # Outputs, figures (empty)
└── 📁 logs/                         # Experiment logs (empty)
```

## What's Installed

✅ PyTorch 2.1.1
✅ Transformers 4.35.2
✅ NumPy, Pandas, Scikit-learn
✅ Matplotlib, Seaborn, Plotly
✅ All custom modules verified

## Quick Start

### Option 1: Jump Right In
```bash
jupyter notebook
# Open: notebooks/01_exploration.ipynb
```

### Option 2: Test First
```bash
# Run a quick test
python -c "from src.data_generation import generate_dataset, SAMPLE_QUESTIONS; \
print(f'Created {len(generate_dataset(SAMPLE_QUESTIONS[:1], 2))} examples')"
```

## Next Steps

### Hour 0: Preparation (doesn't count toward 20h limit)
1. ✅ Environment set up
2. Read through MATS_APPLICATION_RULES.md
3. Review RESEARCH_PLAN.md
4. Familiarize yourself with the code in `src/`

### Hour 1-4: Phase 1 - Exploration
📍 **START HERE**: `notebooks/01_exploration.ipynb`

**Goals**:
- Generate dataset with hints
- Run models and collect responses
- **Manually examine 50-100 examples** ← MOST CRITICAL
- Document patterns

**Key Files**:
- `src/data_generation.py` - Create questions
- `src/model_inference.py` - Run models

### Hour 5-10: Phase 2 - Probe Training
**Goals**:
- Train linear probes per layer
- Compute AUROC + statistical tests
- Compare model types

**Key Files**:
- `src/probe_training.py` - Train probes, permutation tests

### Hour 11-16: Phase 3 - Deep Dive
**Goals**:
- Analyze failure cases
- Run sanity checks
- Compare to baselines

**Key Files**:
- `src/analysis.py` - Qualitative analysis

### Hour 17-22: Phase 4 - Write-up
**Goals**:
- Create visualizations
- Write methodology
- Write executive summary (2 extra hours)

## Important Reminders

### From MATS Guidelines

**DO**:
✅ Start simple (read CoT before training probes)
✅ Look at your data constantly
✅ Be skeptical of results
✅ Compare to baselines
✅ Document everything as you go
✅ Include graphs in executive summary
✅ Be honest about limitations

**DON'T**:
❌ Rely only on cherry-picked examples
❌ Skip manual examination
❌ Train probes without understanding the data
❌ Claim more than evidence supports
❌ Continue doomed projects (pivot if needed)

### Critical Success Factors

1. **Clear Communication**: If Neel can't understand it, it won't pass
2. **Truth-Seeking**: Negative results with good analysis > poorly supported positive results
3. **Good Taste**: Choose interesting problems aligned with Neel's interests
4. **Simplicity First**: Try obvious methods before complex ones

## Research Question Recap

> Can linear probes trained on reasoning model hidden states detect unfaithful hint usage—where the model changes its answer due to a hint without verbalizing that reliance—at rates significantly above chance?

**This aligns with Neel's interests in**:
- Chain-of-thought faithfulness ✅
- Reasoning models ✅
- Model biology ✅
- Applied interpretability ✅

## Time Tracking

Remember to track your time! What counts:
- ✅ Writing code for the project
- ✅ Reading papers relevant to your project
- ✅ Analyzing data/results
- ✅ Writing the main write-up
- ❌ General learning done beforehand
- ❌ Tech setup
- ❌ Waiting for models to run

## Models to Use

**Start small** (testing):
```python
model = ModelWrapper("Qwen/Qwen2.5-0.5B-Instruct")
```

**Then scale up**:
- Reasoning: `Qwen/QwQ-32B-Preview` or `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Baseline: `Qwen/Qwen2.5-7B-Instruct`

## Getting Help

- **MATS Rules**: See `MATS_APPLICATION_RULES.md`
- **Research Plan**: See `RESEARCH_PLAN.md`
- **Quick Commands**: See `QUICK_START.md`
- **Code Examples**: Check docstrings in `src/` files

## GPU Setup (Optional)

⚠️ Note: No GPU detected. You can:
1. Continue on CPU (slower but works for testing)
2. Use Google Colab (free GPU)
3. Rent GPU from RunPod.io or Vast.ai

For Colab setup:
```python
!pip install -r requirements.txt
# Upload your code/notebooks
```

## Final Checklist Before Starting

- [x] Environment verified
- [x] Code modules working
- [ ] Read MATS_APPLICATION_RULES.md
- [ ] Review RESEARCH_PLAN.md
- [ ] Understand research question
- [ ] Set up time tracking
- [ ] Ready to start exploration!

---

## 🚀 You're All Set!

Open `notebooks/01_exploration.ipynb` and start your research journey.

**Remember**: The goal is to learn something interesting and communicate it clearly, whether results are positive or negative!

Good luck! 🎯
