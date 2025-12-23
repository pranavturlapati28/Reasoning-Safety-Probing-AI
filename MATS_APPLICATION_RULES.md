# MATS 10.0 Application Rules & Guidelines
## Neel Nanda's Stream - Summer 2026

---

## KEY DEADLINES
- **Application Due:** Tuesday, December 23rd, 11:59pm PT
- **Late Applications:** Accepted until January 2nd (request extension via form)
- **Decisions Released:** January 12th
- **Exploration Phase:** February 2 - March 6 (5 weeks, online)
- **Research Phase Decisions:** March 12
- **Research Phase:** June 1 - August 21 (12 weeks, in-person in Berkeley)

---

## APPLICATION REQUIREMENTS

### Core Task
- **Spend ~16 hours (max 20)** working on a mechanistic interpretability research problem of your choice
- **Additional 2 hours** for writing the executive summary
- Submit a write-up + executive summary showing what you learned

### What NOT to Count Toward Time Limit
- General prep (paper reading, tutorials) done before choosing a project
- Generic tech setup (renting/setting up cloud GPU)
- Breaks
- Time waiting for training (if doing something else)
- Writing answers to MATS application form

### What DOES Count Toward Time Limit
- Writing code for your project
- Reading papers relevant to your specific project
- Analyzing data/experimental results
- Thinking and planning time
- Writing the main write-up (not executive summary)

---

## RESEARCH INTERESTS (CRITICAL)

### What Neel IS Interested In:
1. **Pragmatic Interpretability** - Clear applications to AGI Safety
2. **Model Biology** - Studying qualitative high-level properties of models
3. **Applied Interpretability** - Rigorously doing useful things with interp
4. **Reasoning Models** - Understanding chain-of-thought, faithfulness
5. **Understanding Weird Behavior** - Debugging unexpected model behavior
6. **Basic Science** - With a higher bar than before

### What Neel is NOT Particularly Interested In:
- **Grokking research**
- **Toy models** (unless great pitch)
- **Most SAE work** (sparse autoencoders)
- **Ambitious interpretability** (complete reverse-engineering)
- Algorithmic tasks
- Interpretability during training (unless great pitch)

### Key Research Philosophy
- **Start simple** - Try obvious methods first (prompting, reading CoT, linear probes)
- **Pragmatism over complexity** - Each piece of complexity should have a reason
- **Truth-seeking** - Negative/inconclusive results with good analysis > poorly supported positive results
- **Avoid cherry-picking** - Need more than a few qualitative examples

---

## APPLICATION FORMAT

### Executive Summary (CRITICAL)
- **First 1-3 pages** of your Google doc
- **Ideal length:** ~1 page including graphs
- **Maximum:** 3 pages, 600 words
- **Must include graphs!**
- Should stand alone and convey key info

### Recommended Executive Summary Structure:
1. What problem are you trying to solve? (Why is it interesting?)
2. What are your high-level takeaways? Most interesting parts?
3. One paragraph + graph per key experiment:
   - What was the experiment?
   - What did you find?
   - Why does this support your takeaways?

### Full Write-Up Requirements:
- Google doc with graphs and sufficient detail
- Should be understandable without reading code
- Code can be included but is optional
- **Make the doc accessible to anyone with the link!**
- Track your time (recommended: use Toggl and include screenshot)

---

## EVALUATION CRITERIA (In Order of Importance)

### 1. Clarity (Top 20% Threshold)
- Can Neel understand your claims, evidence, and reasoning?
- Show enough detail: data generation, prompts, metrics, hyperparameters
- Use bullet points, good graphs, summaries, good structure
- Avoid "illusion of transparency" - explain everything from ground up

### 2. Good Taste
- Choose interesting questions that align with Neel's research interests
- Get meaningful traction on the problem
- Produce compelling results
- **Originality is a big plus**
- Teaching Neel something new is ideal

### 3. Truth-Seeking & Skepticism
- Question your results constantly
- Look for alternative explanations
- Do sanity checks
- **Self-awareness of limitations**
- Avoid overconfidence in shaky results
- **Compare to baselines when applicable**
- **Avoid relying only on cherry-picked qualitative examples** (major red flag)

### 4. Technical Depth & Practicality
- Demonstrate good handle on relevant tools
- Show willingness to write code and run experiments
- Demonstrate understanding vs. blindly following recipes/LLMs
- Knowledge areas: mech interp techniques, working with large models, linear algebra, transformers, coding

### 5. Simplicity
- Bias toward simple, obvious methods first
- Each complexity should have a clear reason
- Pragmatic and focused over showing off

### 6. Prioritization
- Go deep on 1-2 key insights vs. superficial on many things
- Avoid rabbit holes on uninteresting anomalies
- Know when to pivot
- Balance between depth and breadth

### 7. Productivity
- Fast feedback loops
- Notice and fix inefficiency
- Take action or reflect appropriately
- Quality AND speed

### 8. Show Your Work
- Explain your thought process
- Why did you make each decision?
- If stuck: show what you tried, why, and what happened
- Structure write-up to emphasize findings (not chronological)

### 9. Enthusiasm & Curiosity
- Applications that are fun to read get bonus points
- Follow your curiosity productively

---

## COMMON MISTAKES TO AVOID

### Skepticism Failures:
- Not acknowledging limitations
- Pretending negative results are positive (LYING IS NOT OKAY)
- Not checking simple hypotheses before complex ones
- Not sanity-checking if phenomena actually exists
- Using models that are too dumb for the task (e.g., GPT-2)
- Not looking at your data - **READ YOUR DATA!**

### Problem Choice Failures:
- Choosing uninteresting problems outside Neel's interests
- Choosing problems that don't make sense
- Choosing super ambitious/conceptually messy problems
- Pet interests that only appeal to narrow audiences

### Strategy Failures:
- Continuing doomed projects instead of pivoting
- **Can reset 20-hour limit if totally changing direction**

### Misc Failures:
- Poor writing - if executive summary is unclear, application rejected
- LLM-written applications about made-up experiments
- Spreading too thin across many superficial things

---

## RECOMMENDED PROBLEM AREAS

### Model Biology
- Deep dives into weird behavior (self-preservation, confusion, misalignment)
- Chain of thought faithfulness studies
- User models and representation
- Out of context reasoning / emergent misalignment
- Synthetic document fine-tuning effects
- Concept representations (truth, deception, uncertainty)
- Model diffing (before/after fine-tuning)

### Reasoning Models
- CoT faithfulness investigation
- Thought anchors (building on Bogdan et al.)
- Steganography in CoT
- Backtracking behavior
- Editing/resampling CoT

### Circuit Analysis
- Attribution graphs usage and limitations
- Baseline method improvements (probes, reading CoT, observing behavior)
- Automation of hypothesis generation + validation
- Transcoders and cross-layer connections

### Applied Interpretability
- Monitoring with probes (improving probe techniques)
- Analyzing/using CoT for practical tasks
- Conditional steering (steering + probes)
- Abliteration techniques
- Training data attribution

### Basic Science
- Understanding reasoning model internals
- Steering fine-tuning (building on Casademunt et al.)
- SAE basic science (what concepts are learned, Matryoshka SAEs)
- Sanity checking superposition

### Objective Measures
- Eliciting latent knowledge
- Understanding-based downstream tasks

---

## TECHNICAL RECOMMENDATIONS

### Models to Use:
- **Non-reasoning:** Gemma 3, Llama 3.3
- **Reasoning:** Qwen 3 (or Nemotron 49B for better quality)
- **For SAEs:** Gemma 2 with Gemma Scope
- **Avoid:** GPT-2, Mixture of Experts models (unless necessary)

### Libraries & Tools:
- **Small models (≤9B):** TransformerLens
- **Larger models:** nnsight
- **Coding:** Cursor (with AI integration)
- **GPU rental:** runpod.io (or vast.ai if cost-constrained)
- **LLM API:** OpenRouter

### LLM Usage:
- **Strongly encouraged!** LLMs are crucial research tools
- Use for coding, writing, learning, hypothesis generation
- **Recommended:** Cursor for coding, Claude 4.5 Sonnet or Gemini 3 Pro for browser tasks
- Use context effectively - Gemini 3 Pro has 1M context window
- **Responsibility:** Ensure code and writing are high quality
- **LLM-written slop will be rejected**

### Key Resource:
- Use the folder of useful text files for mech interp (600k token default file)
- Contains docs, source code, tutorials, papers, blog posts

---

## WRITING ADVICE

### Focus on Narrative
- Structure around 1-2 most interesting insights
- Don't just dump experiments
- Quality over quantity

### Show Your Work
- Explain WHY you ran each experiment, not just what
- What hypothesis were you testing?
- What were possible outcomes?

### Assume Zero Context
- Define all terms
- Label graphs clearly
- Explain from ground up
- Things obvious to you are new to reader

### Executive Summary Critical
- Must stand alone
- Convey most important takeaways
- Sketch key evidence
- Good graphs are huge plus
- Don't make reader hunt for the point

---

## RESEARCH PROCESS (3 PHASES)

### 1. Exploration (Gain information, build intuition)
- Get hands dirty: read data, try prompts, use SAEs
- Maximize information gain per unit time
- Ask every 30 min: "Have I learned anything? Is this still fruitful?"

### 2. Understanding (Test hypotheses carefully)
- Keep running doc with hypothesis list
- Design clean experiments
- Analyze results
- Track kind of claim: existence proof vs. comparative method evaluation

### 3. Distillation (Communicate clearly)
- **NOT an afterthought!**
- Provide clear evidence for claims
- Show self-awareness of limitations
- Make plausible claims over ambitious ones

---

## PROGRAM STRUCTURE

### Exploration Phase (Top ~34 candidates)
- 5 weeks online: Feb 2 - March 6
- First 3 weeks part-time (preparation)
- Final 2 weeks full-time (research sprint in pairs)
- Stipend: $4.2K
- Admission to research phase based mainly on sprint performance

### Research Phase (Top ~8 candidates)
- 12 weeks in-person in Berkeley: June 1 - Aug 21
- Work in pairs on mech interp paper
- 1.5 hr/week check-ins with Neel
- Stipend: $14.4K + housing support
- Typical outcome: Co-first author paper at top ML venue (NeurIPS/ICLR/ICML)
- Optional 3-12 month extension possible

---

## ELIGIBILITY & LOGISTICS

### Background
- **All backgrounds and experience levels welcome**
- Designed to be meritocratic
- Past scholars: professors, undergrads with no mech interp experience, startup founders, software engineers, researchers with papers

### Location & Visa
- No US location or work authorization required
- Exploration phase: Remote
- Research phase: Encouraged in-person (Berkeley), but can be remote
- Educational program (not employment) - simpler visa process

### Prior Work
- Can submit existing mech interp work (co-first author paper, significant contribution, or high-effort blog post)
- Must include time estimate and contribution description
- **Held to higher standard** than regular applications
- **Much prefer standard application**

---

## IMPORTANT NOTES

### Time Management
- You CAN reset 20-hour limit if completely changing project direction
- Take breaks every 1-2 hours to assess if making progress or in rabbit hole
- Spend max 5 of 12-20 hours on reading papers/tutorials
- Bias toward getting hands dirty with code and experiments

### Existing mech interp work (≤20 hours)
- If done on your own in ≤20 hours but not for application: treat as normal application

### Multiple Applications
- Can apply to multiple MATS mentors
- Receive all research phase offers simultaneously

### LLM Practice
- If new to LLMs for research: **practice before starting official application**
- Pick area/paper and speed-run understanding it
- Your 20 application hours shouldn't be first 20 hours doing research with LLMs

---

## RESOURCES

### Essential Reading
- Neel's blog post: "A Mechanistic Interpretability Analysis of Grokking"
- "Pragmatic Interpretability" post
- 200 Concrete Open Problems in Mechanistic Interpretability
- ARENA tutorials (especially Chapter 1.2, sections 1-3)

### Video Resources
- 3Blue1Brown ML videos and Linear Algebra series
- Neel's YouTube tutorials (intro to transformers, research streams)
- Machine Learning Street Talk podcast interview

### Tools & Libraries
- TransformerLens documentation and tutorials
- nnsight documentation
- Gemma Scope (SAEs for Gemma 2)
- Neuronpedia (SAE latent explorer)

---

## ANTI-PATTERNS

### DO NOT:
- Use emojis (unless explicitly requested)
- Create documentation files proactively
- Rely only on cherry-picked examples
- Overcomplicate when simple methods suffice
- Continue obviously doomed projects
- Submit LLM-written prose without heavy editing
- Apply techniques blindly without understanding
- Work on grokking, toy models, or most SAE work (unless exceptional pitch)

### DO:
- Start with simple, obvious approaches
- Read your data and talk to your model
- Use baselines for comparison
- Show self-awareness of limitations
- Pivot when appropriate
- Track your time
- Include graphs in executive summary
- Make Google doc publicly accessible
- Ask questions if unsure

---

## EVALUATION MINDSET

**Neel's ideal application: One that teaches him something new**

Applications are evaluated holistically on:
1. Did I learn something interesting?
2. Does this show strong research potential?
3. Does this align with my research interests?
4. Is the communication clear and honest?

**A good application is enough for acceptance, regardless of background.**

---

## FINAL TIPS

- If you hear about the research interests and think "that sounds boring, I'm not interested" → Great! Better to know now. Check other MATS mentors.
- If you're not sure whether to apply: Try it! The application process is designed to be educational.
- The exploration phase median participant rates it 1.5x-2.5x counterfactual use of time, even if not accepted to research phase.
- Many exploration-phase-only scholars have gone on to find other mentors or publish papers with Neel's help.

---

**Questions?** Email Neel (contact info in application form)

**Good luck!**
