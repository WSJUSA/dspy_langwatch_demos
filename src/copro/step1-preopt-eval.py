# This script runs a pre-evaluation of the 4.1-nano model with 
# the baseline DSPy module GrammaticalityClassifier against the devset 
# Any false predictions are saved to a CSV file for later use to 
# test the optimized DSPy module.

import os
import sys
import csv
import dspy
import langwatch
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from datasets import load_dataset
import pandas as pd
from optimizers.copro.dspy_grammatically import GrammaticalityClassifier, custom_metric

# Load CoLA dataset (Corpus of Linguistic Acceptability)
# Source https://huggingface.co/datasets/linxinyuan/cola/tree/main
dataset = load_dataset("glue", "cola")
train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()

# Shuffle and truncate a subset of the training split for demonstration
train_df = train_df.sample(frac=1, random_state=23).head(10).reset_index(drop=True)
# Shuffle and truncate a subset of the test split for demonstration
val_df = val_df.sample(frac=1, random_state=23).head(300).reset_index(drop=True)

# Create trainset and devset
trainset = [dspy.Example(sentence=ex['sentence'], label=str(ex['label'])).with_inputs('sentence') for ex in train_df.to_dict(orient='records')]
devset = [dspy.Example(sentence=ex['sentence'], label=str(ex['label'])).with_inputs('sentence') for ex in val_df.to_dict(orient='records')]

# Keys are kept in .env file
load_dotenv()

# Configure DSPy - here using the cheapest model available 4.1-nano from OpenAI
lm = dspy.LM(model="openai/gpt-4.1-nano", api_key=os.environ.get("OPENAI_API_KEY"))
dspy.settings.configure(lm=lm)

# Define evaluation of the test/dev set to see how well the cheapestmodel performs without optimization
evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=10)

# Initialize Langwatch
try:
    langwatch.setup(
        api_key=os.environ.get("LANGWATCH_API_KEY"),
        endpoint_url=os.environ.get("LANGWATCH_ENDPOINT")
    )
# If Langwatch setup fails exit so we do not run llm calls without observability
except Exception as e:
    print(f"LangWatch setup failed: {e}")
    sys.exit(1)  
langwatch.dspy.init(experiment="grammar-4.1-nano-base-eval", optimizer=None, evaluator=evaluator) #<-- pass the evaluator for logging to LangWatch-Evaluations

# Compile classifier module
classifier = GrammaticalityClassifier() #<-- our DSPy module is referenced here

# Run DSPy evaluation and get the results
score, results = evaluator(classifier, metric=custom_metric)

# Each item in eval_result.results is a tuple: (example, prediction, score)
mismatches = [
    {
        "sentence": ex.sentence,
        "true_label": ex.label,
        "predicted_label": pred.label,
        "is_correct": metric_result
    }
    for ex, pred, metric_result in results
    if not metric_result
]

# Extract the LLM mismatched items to a CSV file as a set for optimiized analysis
if mismatches:
    pd.DataFrame(mismatches).to_csv("cola_grammar_errors.csv", index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(mismatches)} error items to cola_grammar_errors.csv")
else:
    print("No errors found in evaluation.")