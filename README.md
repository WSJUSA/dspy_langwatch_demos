# DSPy Optimizer Examples with Langwatch Observability

This repository demonstrates the use of various DSPy optimizers integrated with Langwatch for observability and metrics logging.

## Setup
1. Langwatch requires Python 3.12, does not currently support 3.13

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with the required API keys for OpenAI and Langwatch.

   # OpenAI API Key
   OPENAI_API_KEY="openai-api-key"

   # LangWatch Config
   LANGWATCH_API_KEY="langwatch-project-key"
   LANGWATCH_ENDPOINT="http://localhost:5560" or if hosting https://langwatch.ai/

## Optimizer Examples

- COPRO: See `/src/copro/` and `notebooks/copro-tutorial.ipnyb`

## Tutorials
Each optimizer contains a Jupter Notebook end to end tutorial in notebooks/, as well as python code files implementing the optimizer with Langwatch integration and metrics logging in src/.