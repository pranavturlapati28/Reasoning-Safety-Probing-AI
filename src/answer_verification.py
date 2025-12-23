"""
Answer verification and correction using OpenAI API.

This module verifies the correct answers in generated questions and
updates them if needed.

Usage:
    python -m src.answer_verification --input data/generated_questions.json
"""

import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
import argparse
import time
from tqdm import tqdm
import pandas as pd


class AnswerVerifier:
    """Verify and correct answers in generated questions."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize answer verifier.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o recommended for accuracy)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def verify_answer(
        self,
        question: str,
        given_answer: str,
        category: str
    ) -> Dict:
        """
        Verify if an answer is correct and calculate the right answer.

        Args:
            question: The question text
            given_answer: The answer to verify
            category: Question category (math, logic, etc.)

        Returns:
            Dictionary with verification results
        """
        prompt = f"""Calculate the correct answer for this question.

QUESTION: {question}
CATEGORY: {category}
GIVEN ANSWER (to verify): {given_answer}

Solve this question step-by-step and provide the correct answer.
If the given answer is wrong, provide the corrected answer.
Be precise - for math, include units. For facts, be exact.

Respond in JSON:
{{
  "correct_answer": "the definitive correct answer",
  "given_answer_was_correct": true/false,
  "explanation": "brief explanation of calculation/reasoning",
  "confidence": "high|medium|low"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise expert at math, logic, and factual questions. Calculate correct answers carefully."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Error verifying answer: {e}")
            return {
                "correct_answer": given_answer,  # Keep original on error
                "given_answer_was_correct": None,
                "explanation": f"Error: {str(e)}",
                "confidence": "low"
            }

    def verify_questions_file(
        self,
        filepath: str,
        backup: bool = True,
        rate_limit_delay: float = 0.5
    ) -> Tuple[int, List[Dict]]:
        """
        Verify all answers in a questions JSON file and update if needed.

        Args:
            filepath: Path to questions JSON file
            backup: Whether to create backup before modifying
            rate_limit_delay: Delay between API calls (seconds)

        Returns:
            Tuple of (corrections_made, verification_log)
        """
        # Load questions
        print(f"Loading questions from {filepath}...")
        with open(filepath, 'r') as f:
            questions_data = json.load(f)

        print(f"Loaded {len(questions_data)} questions")

        # Verify and correct
        corrections_made = 0
        verification_log = []

        print("\nVerifying answers with GPT-4o...\n")

        for i, q_data in enumerate(tqdm(questions_data, desc="Verifying")):
            verification = self.verify_answer(
                question=q_data['question'],
                given_answer=q_data['correct_answer'],
                category=q_data['category']
            )

            old_answer = q_data['correct_answer']
            new_answer = verification['correct_answer']

            # Log result
            log_entry = {
                'question_num': i + 1,
                'question': q_data['question'],
                'category': q_data['category'],
                'old_answer': old_answer,
                'new_answer': new_answer,
                'was_correct': verification['given_answer_was_correct'],
                'confidence': verification['confidence'],
                'explanation': verification['explanation']
            }
            verification_log.append(log_entry)

            # Update if different
            if old_answer != new_answer:
                print(f"\n❌ Q{i+1} CORRECTED:")
                print(f"  Question: {q_data['question'][:80]}...")
                print(f"  Old: {old_answer}")
                print(f"  New: {new_answer}")
                print(f"  Reason: {verification['explanation']}")

                q_data['correct_answer'] = new_answer
                corrections_made += 1

            # Rate limiting
            time.sleep(rate_limit_delay)

        # Save results
        if corrections_made > 0:
            if backup:
                # Create backup
                backup_file = filepath.replace('.json', '_backup.json')
                with open(backup_file, 'w') as f:
                    json.dump(questions_data, f, indent=2)
                print(f"\n✅ Original file backed up to: {backup_file}")

            # Save corrected version
            with open(filepath, 'w') as f:
                json.dump(questions_data, f, indent=2)
            print(f"✅ Updated {corrections_made} answers in: {filepath}")
        else:
            print("\n✅ All answers were already correct! No changes needed.")

        return corrections_made, verification_log

    def save_verification_log(
        self,
        log: List[Dict],
        output_path: str
    ):
        """Save verification log to CSV."""
        df = pd.DataFrame(log)
        df.to_csv(output_path, index=False)
        print(f"✅ Verification log saved to: {output_path}")

    def print_summary(
        self,
        total_questions: int,
        corrections_made: int,
        verification_log: List[Dict]
    ):
        """Print summary of verification results."""
        print(f"\n{'='*80}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total questions verified: {total_questions}")
        print(f"Corrections made: {corrections_made}")
        print(f"Accuracy rate: {100*(total_questions-corrections_made)/total_questions:.1f}%")

        if corrections_made > 0:
            print(f"\nCorrected questions:")
            for log in verification_log:
                if log['old_answer'] != log['new_answer']:
                    print(f"\n  Q{log['question_num']}: {log['question'][:80]}...")
                    print(f"    {log['old_answer']} → {log['new_answer']}")

        # Low confidence warnings
        low_confidence = [log for log in verification_log if log['confidence'] == 'low']
        if low_confidence:
            print(f"\n⚠️ Low confidence verifications ({len(low_confidence)} questions):")
            for log in low_confidence:
                print(f"  Q{log['question_num']}: {log['question'][:60]}...")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Verify and correct answers in generated questions"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/generated_questions.json",
        help="Input questions JSON file"
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default="results/answer_verification_log.csv",
        help="Output path for verification log CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (gpt-4o or gpt-4o-mini)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup before modifying"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Create verifier
    verifier = AnswerVerifier(model=args.model)

    # Verify questions
    corrections_made, verification_log = verifier.verify_questions_file(
        filepath=args.input,
        backup=not args.no_backup,
        rate_limit_delay=args.rate_limit
    )

    # Save log
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    verifier.save_verification_log(verification_log, args.output_log)

    # Print summary
    verifier.print_summary(
        total_questions=len(verification_log),
        corrections_made=corrections_made,
        verification_log=verification_log
    )

    print(f"\n✅ Done! Verification complete.")
    print(f"\nTo use in your code:")
    print(f"  from src.answer_verification import AnswerVerifier")
    print(f"  verifier = AnswerVerifier()")
    print(f"  corrections, log = verifier.verify_questions_file('{args.input}')")


if __name__ == "__main__":
    main()
