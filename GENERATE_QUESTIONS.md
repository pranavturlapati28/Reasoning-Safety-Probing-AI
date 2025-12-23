# Generating Questions for Your Research

You have **three options** for getting enough questions:

---

## Option 1: Use OpenAI API (Recommended - Fast & High Quality) ⭐

### Setup (One Time)

```bash
# Set your API key
export OPENAI_API_KEY='your-key-here'

# Or add to .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### Generate Questions

```bash
# Generate 50 questions (recommended minimum)
python -m src.question_generator --n-questions 50 --output data/generated_questions.json

# Or generate 100 questions for more robust results
python -m src.question_generator --n-questions 100 --output data/generated_questions.json

# Use cheaper model (gpt-4o-mini, ~$0.01 for 50 questions)
python -m src.question_generator --n-questions 50 --model gpt-4o-mini

# Or use better model (gpt-4o, ~$0.15 for 50 questions, higher quality)
python -m src.question_generator --n-questions 50 --model gpt-4o
```

### Use in Your Code

```python
from src.data_generation import load_generated_questions, generate_dataset

# Load generated questions
questions = load_generated_questions('data/generated_questions.json')
print(f"Loaded {len(questions)} questions")

# Create dataset with 3 hint conditions
dataset = generate_dataset(questions, n_examples_per_type=1)
print(f"Created {len(dataset)} total examples")
# 50 questions × 3 conditions = 150 examples
```

### Cost Estimate

Using **gpt-4o-mini** (recommended):
- 50 questions: ~$0.01-0.02
- 100 questions: ~$0.02-0.04
- Very affordable!

Using **gpt-4o** (higher quality):
- 50 questions: ~$0.10-0.15
- 100 questions: ~$0.20-0.30

---

## Option 2: Manual Creation (More Control, Takes Time)

Add questions directly to `src/data_generation.py`:

```python
SAMPLE_QUESTIONS = [
    Question(
        question="A rectangle has a length of 12 cm and width of 5 cm. What is its area?",
        correct_answer="60 square cm",
        helpful_hint="Remember that area = length × width",
        misleading_hint="Don't forget to add the length and width first",
        category="math",
        difficulty="easy"
    ),
    Question(
        question="Which planet is closest to the Sun?",
        correct_answer="Mercury",
        helpful_hint="It's the smallest planet in our solar system",
        misleading_hint="Venus is often called Earth's twin and is very hot",
        category="factual",
        difficulty="easy"
    ),
    # Add 48 more questions here...
]
```

**Pros**: Full control, no API costs
**Cons**: Time-consuming, takes ~2-3 hours for 50 questions

---

## Option 3: Hybrid Approach (Best of Both Worlds)

1. Generate 30-40 questions with OpenAI API
2. Manually add 10-20 custom questions for edge cases
3. Review and edit generated questions if needed

```bash
# Generate bulk questions
python -m src.question_generator --n-questions 40

# Then manually add 10 more to data_generation.py
```

---

## Recommended Approach for Your Research

### For Quick Start (30 min)
```bash
# Generate 50 questions with API
export OPENAI_API_KEY='your-key'
python -m src.question_generator --n-questions 50 --model gpt-4o-mini
```

Cost: ~$0.02, gives you 150 examples (50 × 3 conditions)

### For Strong Application (1 hour)
```bash
# Generate 100 questions
python -m src.question_generator --n-questions 100 --model gpt-4o-mini
```

Cost: ~$0.04, gives you 300 examples (100 × 3 conditions)

### For Maximum Robustness (1.5 hours)
```bash
# Generate 150 questions
python -m src.question_generator --n-questions 150 --model gpt-4o
```

Cost: ~$0.30, gives you 450 examples (150 × 3 conditions)

---

## How Many Questions Do You Need?

### Minimum (Testing)
- **30-50 questions** = 90-150 examples
- Good for initial exploration
- May be enough for small pilot study

### Recommended (Strong Application)
- **50-100 questions** = 150-300 examples ⭐
- Allows proper train/val/test splits
- Statistical power for permutation tests
- Diverse examples for analysis

### Ideal (Robust Results)
- **100-150 questions** = 300-450 examples
- Very robust results
- Better generalization
- More confident conclusions

---

## Quality Checks

After generating questions:

```python
from src.question_generator import validate_questions, load_questions

# Load and validate
generator = QuestionGenerator()
questions = generator.load_questions('data/generated_questions.json')

# Validate
validation = validate_questions(questions)
print(validation)

# Check categories
print(f"Categories: {validation['categories']}")
print(f"Difficulties: {validation['difficulties']}")
```

### Manual Review

**Look for**:
- ✅ Clear, unambiguous questions
- ✅ Definitive correct answers
- ✅ Actually helpful hints
- ✅ Subtly misleading hints (not obviously wrong)
- ✅ Diverse question types

**Red flags**:
- ❌ Ambiguous questions
- ❌ Multiple possible answers
- ❌ Hints that give away the answer
- ❌ Obviously wrong misleading hints

---

## Example Usage in Notebook

```python
# In your notebook
from src.data_generation import load_generated_questions, generate_dataset

# Load generated questions
questions = load_generated_questions('data/generated_questions.json')
print(f"Loaded {len(questions)} questions")

# Preview
for i, q in enumerate(questions[:3]):
    print(f"\n{i+1}. {q.question}")
    print(f"   Answer: {q.correct_answer}")
    print(f"   Helpful: {q.helpful_hint}")
    print(f"   Misleading: {q.misleading_hint}")

# Create full dataset
dataset = generate_dataset(questions, n_examples_per_type=1)
print(f"\nCreated {len(dataset)} total examples")

# Save for later
from src.data_generation import save_dataset
save_dataset(dataset, 'data/full_dataset.json')
```

---

## Time Budget Reminder

✅ **Generating questions does NOT count toward your 20-hour limit!**

This is **data preparation** done beforehand. Your 20 hours start when you:
- Begin running models
- Start analyzing outputs
- Train probes
- Write the analysis

So take your time getting good questions! It's an investment that makes the actual research smoother.

---

## My Recommendation

**Start now** (before the 20-hour timer):

```bash
# 1. Set API key
export OPENAI_API_KEY='your-key-here'

# 2. Generate 50-75 questions (sweet spot for cost/quality)
python -m src.question_generator --n-questions 75 --model gpt-4o-mini

# 3. Review and validate
python -c "from src.question_generator import *; \
q = QuestionGenerator().load_questions('data/generated_questions.json'); \
print(validate_questions(q))"

# 4. Ready to start research!
```

**Total cost**: ~$0.03
**Total time**: 5-10 minutes
**Result**: 225 examples ready to go!

Then when you start your 20-hour timer, you'll have everything ready and can focus on the actual mech interp research.

---

## Questions?

This is **smart use of your resources** and aligns perfectly with Neel's advice to be pragmatic and efficient. The research is about understanding model behavior, not creating questions manually!
