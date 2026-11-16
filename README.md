# Model Race

An experimental Python CLI project comparing **TF-IDF + Logistic Regression**, a **Cloud LLM (Gemini or Groq)**, and a **Local LLM (Ollama)** on the exact same 100-review test split from 500 labelled product reviews.

## Objective

Understand and measure when a small, inexpensive traditional ML classifier can match or outperform a Large Language Model in accuracy, F1-score, cost, and inference speed.

## Technology Stack

- Python 3.11+
- `scikit-learn` (TF-IDF + Logistic Regression & evaluation metrics)
- `pandas` (dataset inspection and validation)
- `google-genai` / `httpx` (Gemini, Groq, and Ollama inference via Assignment 6 `generate()` pattern)
- `python-dotenv` (configuration)
- `uv` (dependency and environment management)
- `pytest` (unit tests)

## Repository Structure

```text
model-race/
├── data/reviews.csv
├── results/llm_cache.json
├── race.py
├── test_race.py
├── results.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
└── .env.example