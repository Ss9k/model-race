# Model Race — Experimental Results

| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |
|---|---|---|---|---|---|---|
| scikit-learn (TF-IDF + Logistic Regression) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.000000 | 0.0117 |
| Cloud LLM (groq: openai/gpt-oss-20b) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.001390 | 429.73 (from prior live run; 100% cached) |
| Ollama (qwen3:8b) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.000000 (local hardware & electricity not free) | 2448.13 (from prior live run; 100% cached) |

### Run Notes

- Test set: 100 stratified reviews (random_state=42).
- Cloud LLM (openai/gpt-oss-20b) usage: 12257 input tokens, 1570 output tokens; out-of-pocket free-tier cost = $0.00; paid list estimate uses documented paid list price ($0.075/1M in, $0.3/1M out).
- Cloud LLM API calls made this run: 0 | Served from cache: 100 | Failures: 0.
- Ollama (qwen3:8b) live calls this run: 0 | Served from cache: 100 | Failures: 0.

### Which Model Would I Ship?

For this 500-review sentiment classification task, I would ship **scikit-learn (TF-IDF + Logistic Regression)**. It achieved an F1-score of 1.0000 and accuracy of 1.0000, matching or staying within 0.0000 F1 of the best-performing LLM (scikit-learn (TF-IDF + Logistic Regression) at 1.0000 F1) while costing $0.00 in API fees and predicting in 0.0117 ms per review. The Cloud LLM (Cloud LLM (groq: openai/gpt-oss-20b)) adds network latency (429.73 (from prior live run; 100% cached) ms/review), external API dependencies, and paid-tier cost ($0.001390), while Ollama (Ollama (qwen3:8b)) requires local GPU/CPU overhead (2448.13 (from prior live run; 100% cached) ms/review) without a meaningful accuracy gain. On well-structured, narrow binary text classification like product reviews, TF-IDF + Logistic Regression delivers the best balance of reliability, speed, and zero marginal inference cost.

### Class Balance and Accuracy

- **Positive reviews:** 250 (50.00%)
- **Negative reviews:** 250 (50.00%)
- **Dataset balance:** The dataset of 500 reviews is **balanced**.

Accuracy can be misleading on imbalanced datasets because a trivial classifier that always predicts the majority class can score high accuracy without learning useful patterns (for example, predicting only positive on a 90/10 dataset gives 90% accuracy despite 0% recall on negative reviews). In this dataset, the split is 50.00% positive vs. 50.00% negative (balanced), so accuracy aligns closely with F1-score here—yet reporting precision, recall, and F1 alongside accuracy remains essential to verify performance on both classes.
