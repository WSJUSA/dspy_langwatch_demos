# COPRO Optimizer Example (DSPy + LangWatch)

This directory demonstrates using the DSPy COPRO optimizer for grammaticality classification, with LangWatch for experiment tracking.

## Files

- `dspy_grammatically.py` — Defines the DSPy signature and module for grammaticality classification.
- `step1-preopt-eval.py` — Runs baseline evaluation of the classifier and saves errors to `cola_grammar_errors.csv`.
- `step2-opt-copro.py` — Runs the COPRO optimizer to improve the prompt and saves the optimized module to `grammatically_optimized.json`.
- `step3-postopt-eval-errors.py` — Evaluates the optimized classifier on the error set.
- `cola_grammar_errors.csv` — Example CSV of sentences the baseline model misclassified.
- `grammatically_optimized.json` — Saved optimized DSPy module after COPRO.
- `copro-tutorial.ipynb` — Jupyter notebook tutorial walking through the full process interactively.

## Usage

1. Install dependencies (see project root `requirements.txt`).
2. Set your OpenAI and LangWatch API keys in a `.env` file.
3. Run the scripts in order:
   - `python step1-preopt-eval.py`
   - `python step2-opt-copro.py`
   - `python step3-postopt-eval-errors.py`

Or, open `notebooks/copro-tutorial.ipynb` for an interactive walkthrough.