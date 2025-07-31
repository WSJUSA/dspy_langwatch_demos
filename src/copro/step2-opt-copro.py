# This script runs the DSPy COPRO optimizer on the DSPy module GrammaticalityClassifier
# and saves the optimized module to a JSON file.

import os
import sys
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


load_dotenv()

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4.1-nano", api_key=os.environ.get("OPENAI_API_KEY"))
dspy.settings.configure(lm=lm)

# Configure optimizer
# COPRO generates new prompt va, specify a higher model for this - here we use gpt-4o-mini
prompt_lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

optimizer = dspy.COPRO(
    prompt_model=prompt_lm,
    metric=custom_metric,
    breadth=3,  #<-- number of prompts to generate
    depth=1,    #<-- number of iterations (iteration = one prompt variation x tainset size)
    init_temperature=1.4
)

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
langwatch.dspy.init(experiment="grammar-copro-4.1-nano", optimizer=optimizer)

# Compile e.g. run the DSPy optimizer on the DSPy module
compiled_module = optimizer.compile(GrammaticalityClassifier(), trainset=trainset, eval_kwargs={"num_threads": 4}) #<-- our DSPy module is referenced here

# Save the optimized module (GrammaticalityClassifier) to a json file
compiled_module.save("grammatically_optimized.json")