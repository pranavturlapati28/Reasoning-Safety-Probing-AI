# Detailed Research Plan: Detecting Unfaithful Hint Usage

## Time Budget: 20 hours + 2 for executive summary

---

## Phase 1: Exploration & Data Collection (4 hours)

### Hour 1: Dataset Generation
**Tasks**:
- [ ] Create 20-30 questions with helpful and misleading hints
- [ ] Focus on domains: math, logic, factual knowledge
- [ ] Ensure hints are clearly helpful or misleading
- [ ] Generate prompts in three conditions: no hint, helpful hint, misleading hint

**Output**:
- `data/questions.json` - Question bank
- `data/dataset.json` - Full dataset with all conditions

### Hours 2-3: Model Inference
**Tasks**:
- [ ] Load models (start with Qwen2.5-0.5B for testing)
- [ ] Run inference on dataset
- [ ] Extract hidden states from key layers
- [ ] Save responses and hidden states

**Models to run**:
1. Qwen2.5-0.5B-Instruct (testing)
2. Then scale to: QwQ-32B or DeepSeek-R1-Distill-7B
3. Baseline: Qwen2.5-7B-Instruct

**Output**:
- `data/model_responses.json` - All model outputs
- `data/hidden_states.npz` - Extracted activations

### Hour 4: Manual Examination
**Tasks**:
- [ ] Read through 50-100 examples manually
- [ ] Note when answers change due to hints
- [ ] Note when hints are verbalized in CoT
- [ ] Identify clear cases of "unfaithful" behavior
- [ ] Document patterns in notebook

**Key Questions to Answer**:
1. Do models change answers based on hints? (How often?)
2. Do they verbalize hint usage? (Always, sometimes, never?)
3. What does unfaithful behavior look like qualitatively?
4. Are there differences between model types?

**Output**:
- Documented observations in `notebooks/01_exploration.ipynb`
- Initial estimate of unfaithfulness rate

---

## Phase 2: Probe Training & Analysis (6 hours)

### Hour 5: Data Preparation
**Tasks**:
- [ ] Create labels for unfaithful vs faithful examples
- [ ] Organize hidden states by layer
- [ ] Split into train/val/test sets (stratified)
- [ ] Verify data quality and balance

**Label Definition**:
- **Unfaithful (1)**: Answer changed due to hint BUT hint not verbalized in CoT
- **Faithful (0)**: Either no change OR hint verbalized if change occurred

### Hours 6-8: Probe Training
**Tasks**:
- [ ] Train linear probes for each layer
- [ ] Cross-validate regularization parameter
- [ ] Identify best-performing layers
- [ ] Compare across model types

**Layers to Focus On**:
- Early layers (0, 1, 2)
- Middle layers (half-way point)
- Late layers (last 3 layers)
- All layers for comprehensive analysis

**Metrics to Track**:
- AUROC (primary metric)
- Accuracy
- Precision/Recall
- Calibration

**Output**:
- `results/probe_performance_by_layer.csv`
- Trained probe models saved

### Hours 9-10: Statistical Validation
**Tasks**:
- [ ] Run permutation tests (1000 permutations)
- [ ] Calculate p-values for AUROC
- [ ] Compare reasoning vs instruction-tuned models
- [ ] Test significance of differences

**Comparisons**:
1. Probe AUROC vs. chance (0.5)
2. Reasoning model vs. instruction-tuned model
3. Best layer vs. other layers

**Output**:
- `results/statistical_tests.json`
- Significance claims with p-values

---

## Phase 3: Deep Dive & Validation (6 hours)

### Hours 11-12: Failure Analysis
**Tasks**:
- [ ] Identify false positives (faithful predicted as unfaithful)
- [ ] Identify false negatives (unfaithful predicted as faithful)
- [ ] Read examples to understand patterns
- [ ] Document what probe might actually be learning

**Investigate**:
- Is probe detecting correctness instead of unfaithfulness?
- Is probe detecting hint presence vs. hint influence?
- Are there systematic patterns in errors?

### Hours 13-14: Sanity Checks
**Tasks**:
- [ ] Train probe to detect hint presence (control)
- [ ] Train probe to detect answer correctness (control)
- [ ] Compare to probe detecting unfaithfulness
- [ ] Test generalization to new hint types/formats

**Critical Questions**:
- Does unfaithfulness probe learn something different from these controls?
- Does it generalize beyond training distribution?

### Hours 15-16: Baseline Comparisons
**Tasks**:
- [ ] LLM-based classification of hint verbalization
- [ ] Compare probe AUROC to LLM classification
- [ ] Simple rule-based methods (keyword matching)
- [ ] Evaluate cost/benefit of each method

**Baselines**:
1. GPT-4 reading CoT and classifying
2. Keyword matching for hint terms
3. Answer change detection only

**Output**:
- `results/baseline_comparison.csv`
- Analysis of when probe adds value

---

## Phase 4: Analysis & Visualization (4 hours)

### Hours 17-18: Create Key Visualizations
**Tasks**:
- [ ] AUROC by layer (reasoning vs instruct models)
- [ ] ROC curves for best probes
- [ ] Confusion matrices
- [ ] Example predictions with confidence scores
- [ ] Calibration plots

**Figures to Create**:
1. `fig1_auroc_by_layer.png` - Main result
2. `fig2_roc_curves.png` - Performance comparison
3. `fig3_example_predictions.png` - Qualitative examples
4. `fig4_confusion_matrix.png` - Error analysis

### Hours 19-20: Main Write-Up
**Tasks**:
- [ ] Write methodology section
- [ ] Document all experiments run
- [ ] Create results tables
- [ ] Write analysis of findings
- [ ] Note limitations clearly

**Structure**:
1. Introduction & Research Question
2. Methodology
3. Results
4. Analysis
5. Limitations
6. Next Steps

---

## Phase 5: Executive Summary (+2 hours)

### Hours 21-22: Polish & Summarize
**Tasks**:
- [ ] Write 1-page executive summary
- [ ] Include 3-5 key graphs
- [ ] Highlight most interesting findings
- [ ] State limitations honestly
- [ ] Explain implications

**Executive Summary Format**:
1. Research Question (2-3 sentences)
2. Key Findings (bullet points)
3. Evidence (1 paragraph + graph per finding)
4. Limitations (1 paragraph)
5. Implications (2-3 sentences)

---

## Deliverables Checklist

### Code & Data
- [ ] Clean, documented code in `src/`
- [ ] Notebooks with clear narrative
- [ ] Dataset and model responses saved
- [ ] Trained probes saved

### Analysis
- [ ] Quantitative results (AUROC, p-values)
- [ ] Qualitative examples
- [ ] Failure analysis
- [ ] Baseline comparisons

### Visualizations
- [ ] AUROC by layer
- [ ] ROC curves
- [ ] Confusion matrices
- [ ] Example predictions
- [ ] Any other relevant plots

### Write-Up
- [ ] Executive summary (1-3 pages, includes graphs)
- [ ] Full methodology
- [ ] Results section
- [ ] Limitations clearly stated
- [ ] Code/data references

---

## Potential Pivots

### If unfaithfulness is rare (<5% of cases):
- Focus on understanding the rare cases
- Investigate what makes hints more likely to be unfaithful
- Study differences in how models handle edge cases

### If probes don't work (AUROC ~0.5):
- Investigate why (confounds? insufficient signal?)
- Try alternative methods (attention patterns, activation differences)
- Document what this tells us about hint processing

### If results are messy/ambiguous:
- Be honest about uncertainty
- Focus on what we CAN conclude
- Document open questions for future work

---

## Success Criteria

### Minimum Viable Application:
- Clear experimental design
- Honest analysis of results (positive or negative)
- Good communication of findings
- Evidence of skepticism and sanity checks

### Strong Application:
- Interesting findings (positive or negative)
- Clear evidence for claims
- Insightful failure analysis
- Novel observations about model behavior

### Excellent Application:
- Teach Neel something new
- Strong evidence + clear writing
- Thoughtful analysis of mechanisms
- Practical implications identified

---

## Risk Mitigation

### Risk: Models don't show unfaithful behavior
**Mitigation**: Make this the finding! Investigate why.

### Risk: Not enough time for all models
**Mitigation**: Prioritize one reasoning model + baseline.

### Risk: Probes don't work
**Mitigation**: Analyze failure modes, try alternatives.

### Risk: Results are boring
**Mitigation**: Focus on understanding WHY they're boring.

---

## Notes & Reminders

- **Start simple**: Read CoT before training probes
- **Document everything**: Write as you go
- **Pivot if needed**: Don't stick with doomed approaches
- **Be honest**: Negative results are fine
- **Time management**: Set timers, check progress hourly
- **Ask for help**: Use LLMs, check papers, ask questions

**Most Important**: Have fun and learn something!
