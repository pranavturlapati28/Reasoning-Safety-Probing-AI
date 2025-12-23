# Quick Start Guide

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (if using OpenAI/Anthropic for analysis)
echo "OPENAI_API_KEY=your_key_here" > .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env

# 3. Start Jupyter
jupyter notebook
```

## Workflow

### Day 1: Exploration (4 hours)
1. Open `notebooks/01_exploration.ipynb`
2. Generate dataset
3. Run small model for testing
4. **Manually examine outputs** ← MOST IMPORTANT
5. Document patterns

### Day 2: Probe Training (6 hours)
1. Open `notebooks/02_probe_training.ipynb`
2. Prepare data and labels
3. Train probes per layer
4. Run statistical tests
5. Analyze best layers

### Day 3: Deep Dive (6 hours)
1. Open `notebooks/03_analysis.ipynb`
2. Investigate failure cases
3. Run sanity checks
4. Compare to baselines
5. Create visualizations

### Day 4: Write-up (4+2 hours)
1. Compile results
2. Create final figures
3. Write full methodology
4. Write executive summary
5. Polish and submit

## Key Commands

```python
# Generate dataset
from src.data_generation import generate_dataset, SAMPLE_QUESTIONS
dataset = generate_dataset(SAMPLE_QUESTIONS, n_examples_per_type=10)

# Load model
from src.model_inference import ModelWrapper
model = ModelWrapper("Qwen/Qwen2.5-0.5B-Instruct")

# Run inference
result = model.generate_with_hidden_states(
    prompt="Your question here",
    max_new_tokens=256,
    extract_layers=[0, -1]
)

# Train probe
from src.probe_training import LinearProbe
probe = LinearProbe(input_dim=hidden_states.shape[1])
results = probe.train(hidden_states, labels)

# Evaluate
eval_results = probe.evaluate(test_hidden_states, test_labels)
print(f"AUROC: {eval_results['auroc']:.3f}")

# Permutation test
from src.probe_training import permutation_test
perm_results = permutation_test(y_true, y_prob, n_permutations=1000)
print(f"P-value: {perm_results['p_value']:.4f}")
```

## Models to Use

Start with small model for testing:
```python
model = ModelWrapper("Qwen/Qwen2.5-0.5B-Instruct")
```

Then scale up to:
```python
# Reasoning models
model = ModelWrapper("Qwen/QwQ-32B-Preview")
model = ModelWrapper("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# Baseline
model = ModelWrapper("Qwen/Qwen2.5-7B-Instruct")
```

## GPU Setup

If using RunPod or Vast.ai:

```bash
# SSH into instance
ssh -p PORT user@host

# Install dependencies
pip install -r requirements.txt

# Start Jupyter with port forwarding
jupyter notebook --no-browser --port=8888

# On local machine:
ssh -N -L 8888:localhost:8888 -p PORT user@host
# Then open localhost:8888 in browser
```

## Time Tracking

Install Toggl or use simple timer:

```python
import time
start_time = time.time()

# ... do work ...

elapsed = (time.time() - start_time) / 3600
print(f"Time spent: {elapsed:.2f} hours")
```

## Critical Checkpoints

### After Hour 4 (Exploration):
✓ Do you have clear examples of hint influence?
✓ Can you identify unfaithful cases manually?
✓ Is there enough signal to proceed?

**If NO**: Pivot or refine dataset

### After Hour 10 (Probe Training):
✓ Are probes significantly above chance?
✓ Do permutation tests show p < 0.05?
✓ Is there a best layer?

**If NO**: Investigate why, document limitations

### After Hour 16 (Deep Dive):
✓ Do you understand failure modes?
✓ Have you run sanity checks?
✓ Are baselines compared?

**If NO**: Add more analysis before writing

## Common Issues

### Model won't load
- Try smaller model first
- Check GPU memory: `nvidia-smi`
- Use `load_in_8bit=True`

### Probes perform poorly
- Check label distribution (balanced?)
- Try different layers
- Verify labels are correct
- Compare to random baseline

### Running out of time
- Focus on one model type
- Reduce dataset size
- Skip less critical experiments
- Document what you skipped

## Emergency Pivots

### If unfaithfulness is rare:
→ Study the rare cases in depth
→ Compare models on edge cases

### If probes don't work:
→ Document why they fail
→ Try simpler baselines
→ Analyze what confounds exist

### If results are negative:
→ That's a valid finding!
→ Focus on understanding why
→ Be thorough in analysis

## Final Checklist

Before submitting:
- [ ] Executive summary is clear and standalone
- [ ] Key graphs are included
- [ ] Limitations are stated honestly
- [ ] Code is documented
- [ ] Methodology is complete
- [ ] Time tracking is included
- [ ] Google doc is publicly accessible

## Help & Resources

- MATS Application Rules: `MATS_APPLICATION_RULES.md`
- Detailed Plan: `RESEARCH_PLAN.md`
- Example Applications: See MATS_APPLICATION_RULES.md

**Remember**: Focus on clarity, honesty, and learning something interesting!
