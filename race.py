import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

load_dotenv()

RANDOM_STATE = 42
TEST_SIZE = 0.20

CSV_PATH = Path(os.getenv("CSV_PATH", "data/reviews.csv"))
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "review")
LABEL_COLUMN = os.getenv("LABEL_COLUMN", "sentiment")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
CLOUD_REQUEST_DELAY = float(os.getenv("CLOUD_REQUEST_DELAY", "2.1"))

CACHE_PATH = Path("results/llm_cache.json")
RESULTS_PATH = Path("results.md")

PROMPT_TEMPLATE = (
    "You are a sentiment classification system. Classify the following review as either "
    "positive or negative. Return exactly one word: positive or negative. "
    "Do not provide any explanation.\n\n"
    "Review: {review_text}"
)
PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]

# Documented paid list prices (USD per 1M tokens: (input, output))
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

UsageCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# 1. Dataset Loading, Inspection & Validation
# ---------------------------------------------------------------------------


def load_and_validate_dataset(
    csv_path: Path,
    text_col: str,
    label_col: str,
    expected_rows: int | None = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Place the 500-review CSV there and rerun."
        )

    df = pd.read_csv(csv_path, encoding="utf-8")

    print("\n=== Dataset Inspection ===")
    print(f"File: {csv_path}")
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    print("Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' not found in {list(df.columns)}. "
            "Configure TEXT_COLUMN / LABEL_COLUMN in .env."
        )

    missing_text = int(df[text_col].isna().sum())
    missing_label = int(df[label_col].isna().sum())
    print(f"Missing values -> {text_col}: {missing_text}, {label_col}: {missing_label}")

    if missing_text > 0 or missing_label > 0:
        raise ValueError("Dataset contains missing values in required columns.")

    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    if (df[text_col] == "").any():
        raise ValueError("Dataset contains blank review texts.")

    duplicate_count = int(df.duplicated(subset=[text_col]).sum())
    print(f"Duplicate review texts: {duplicate_count}")

    unique_labels = set(df[label_col].unique())
    if not unique_labels.issubset({"positive", "negative"}):
        raise ValueError(
            f"Labels must only be 'positive' or 'negative'. Found: {sorted(unique_labels)}"
        )

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Expected exactly {expected_rows} usable reviews, found {len(df)}."
        )

    total = len(df)
    pos_count = int((df[label_col] == "positive").sum())
    neg_count = int((df[label_col] == "negative").sum())
    pos_pct = (pos_count / total) * 100.0
    neg_pct = (neg_count / total) * 100.0
    is_balanced = abs(pos_pct - neg_pct) <= 10.0
    balance_label = "balanced" if is_balanced else "imbalanced"

    print(f"Positive reviews: {pos_count} ({pos_pct:.2f}%)")
    print(f"Negative reviews: {neg_count} ({neg_pct:.2f}%)")
    print(f"Class distribution: {balance_label.upper()}")

    stats = {
        "total": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "is_balanced": is_balanced,
        "balance_label": balance_label,
        "duplicate_count": duplicate_count,
    }
    return df, stats


def split_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], list[str], list[str], list[str]]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(),
        df[label_col].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col].tolist(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Model 1: Traditional ML (TF-IDF + Logistic Regression)
# ---------------------------------------------------------------------------


def run_traditional_ml(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
) -> tuple[list[str], float, TfidfVectorizer]:
    # Fit TF-IDF strictly on training data only to prevent data leakage.
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # Measure inference speed separately from dataset loading and model training.
    start = time.perf_counter()
    X_test_vec = vectorizer.transform(X_test)
    predictions = clf.predict(X_test_vec).tolist()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return predictions, elapsed_ms / len(X_test), vectorizer


# ---------------------------------------------------------------------------
# 3. Assignment 6 `generate()` Pattern (Groq + Ollama)
# ---------------------------------------------------------------------------


async def generate(
    prompt: str,
    provider: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    provider = provider.lower().strip()

    if provider == "groq":
        async for chunk in generate_with_groq(prompt, on_usage):
            yield chunk
    elif provider == "ollama":
        async for chunk in generate_with_ollama(prompt, on_usage):
            yield chunk
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def generate_with_groq(
    prompt: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not configured in .env.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"
        payload["include_reasoning"] = False

    timeout = httpx.Timeout(REQUEST_TIMEOUT)
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream("POST", url, headers=headers, json=payload) as response,
    ):
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: ") :].strip()
            if raw == "[DONE]":
                break

            data = json.loads(raw)
            choices = data.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]

            usage = data.get("usage") or (data.get("x_groq") or {}).get("usage")
            if usage and on_usage:
                on_usage(
                    {
                        "model": data.get("model", GROQ_MODEL),
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    }
                )


async def generate_with_ollama(
    prompt: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "think": False}
    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line:
                continue

            data = json.loads(line)
            if data.get("response"):
                yield data["response"]

            if data.get("done") and on_usage:
                on_usage(
                    {
                        "model": data.get("model", OLLAMA_MODEL),
                        "input_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                    }
                )


# ---------------------------------------------------------------------------
# 4. Response Normalization & JSON Cache
# ---------------------------------------------------------------------------


def normalize_sentiment(raw_text: str) -> str | None:
    if not raw_text:
        return None
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"^[\s\"'`.,!?:;()]+|[\s\"'`.,!?:;()]+$", "", cleaned)
    if cleaned in {"positive", "negative"}:
        return cleaned
    return None


def build_cache_key(
    provider: str,
    model_name: str,
    review_text: str,
    prompt_hash: str = PROMPT_HASH,
) -> str:
    raw = f"{provider.lower().strip()}|{model_name.strip()}|{prompt_hash}|{review_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path = CACHE_PATH) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, Any], cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    os.replace(tmp_path, cache_path)


# ---------------------------------------------------------------------------
# 5. LLM Runner with Caching, Error Handling & Token Tracking
# ---------------------------------------------------------------------------


async def call_llm_once(
    review_text: str,
    provider: str,
    max_retries: int = 5,
) -> tuple[str, int, int, float, bool]:
    prompt = PROMPT_TEMPLATE.format(review_text=review_text)

    for attempt in range(1, max_retries + 1):
        usage_info: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

        def on_usage(info: dict[str, Any]) -> None:
            usage_info["input_tokens"] = int(info.get("input_tokens") or 0)
            usage_info["output_tokens"] = int(info.get("output_tokens") or 0)

        chunks: list[str] = []
        start = time.perf_counter()
        try:
            async for chunk in generate(prompt, provider, on_usage=on_usage):
                chunks.append(chunk)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_output = "".join(chunks)
            normalized = normalize_sentiment(raw_output)
            if normalized is None:
                raise ValueError(f"Unexpected LLM output: {raw_output!r}")

            in_tok = usage_info["input_tokens"]
            out_tok = usage_info["output_tokens"]
            used_fallback = False
            if in_tok == 0 or out_tok == 0:
                # Clearly disclosed fallback estimate (~4 chars/token) if metadata missing
                in_tok = in_tok or math.ceil(len(prompt) / 4)
                out_tok = out_tok or max(1, math.ceil(len(raw_output) / 4))
                used_fallback = True

            return normalized, in_tok, out_tok, latency_ms, used_fallback

        except Exception as exc:
            err_msg = str(exc).lower()
            is_rate_limit = (
                "429" in err_msg
                or "rate limit" in err_msg
                or "rate_limit" in err_msg
                or "too many requests" in err_msg
            )
            if is_rate_limit and attempt < max_retries:
                retry_after = None
                if isinstance(exc, httpx.HTTPStatusError):
                    header_val = exc.response.headers.get("retry-after")
                    if header_val and header_val.isdigit():
                        retry_after = int(header_val) + 1
                backoff = retry_after or min(65, 10 * attempt)
                print(
                    f"    Rate limit hit; waiting {backoff}s before retry ({attempt}/{max_retries})..."
                )
                await asyncio.sleep(backoff)
                continue
            raise

    raise RuntimeError("Unreachable retry state")


async def run_llm_classification(
    reviews: list[str],
    provider: str,
    model_name: str,
    cache: dict[str, Any],
    cache_path: Path = CACHE_PATH,
    request_delay: float = 0.0,
) -> dict[str, Any]:
    predictions: list[str | None] = []
    live_latencies_ms: list[float] = []
    stored_latencies_ms: list[float] = []
    cache_lookup_latencies_ms: list[float] = []
    failures: list[str] = []

    live_calls = 0
    cache_hits = 0
    total_input_tokens = 0
    total_output_tokens = 0
    fallback_token_count = 0

    for idx, review_text in enumerate(reviews, start=1):
        key = build_cache_key(provider, model_name, review_text)

        lookup_start = time.perf_counter()
        entry = cache.get(key)
        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        if entry and entry.get("prediction") in {"positive", "negative"}:
            predictions.append(entry["prediction"])
            cache_hits += 1
            cache_lookup_latencies_ms.append(lookup_ms)
            total_input_tokens += int(entry.get("input_tokens", 0))
            total_output_tokens += int(entry.get("output_tokens", 0))
            if entry.get("latency_ms") is not None:
                stored_latencies_ms.append(float(entry["latency_ms"]))
            continue

        try:
            pred, in_tok, out_tok, latency_ms, used_fallback = await call_llm_once(
                review_text, provider
            )
            live_calls += 1
            live_latencies_ms.append(latency_ms)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            if used_fallback:
                fallback_token_count += 1

            predictions.append(pred)
            cache[key] = {
                "provider": provider,
                "model": model_name,
                "prompt_hash": PROMPT_HASH,
                "review_preview": review_text[:80],
                "prediction": pred,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": round(latency_ms, 2),
                "token_estimated": used_fallback,
            }
            save_cache(cache, cache_path)

            if idx % 20 == 0 or idx == len(reviews):
                print(f"  [{provider}/{model_name}] Processed {idx}/{len(reviews)} reviews...")

            if request_delay > 0 and idx < len(reviews):
                await asyncio.sleep(request_delay)

        except Exception as exc:
            err = f"Review #{idx}: {type(exc).__name__}: {exc}"
            failures.append(err)
            predictions.append(None)
            print(f"  [ERROR] {provider}/{model_name} failed on {err}")
            if any(
                token in str(exc).lower()
                for token in ("not configured", "400", "401", "403", "404", "connect")
            ):
                print(f"  Aborting remaining {provider} calls due to setup/model/connection error.")
                break

    return {
        "provider": provider,
        "model": model_name,
        "predictions": predictions,
        "live_calls": live_calls,
        "cache_hits": cache_hits,
        "failures": failures,
        "live_latencies_ms": live_latencies_ms,
        "stored_latencies_ms": stored_latencies_ms,
        "cache_lookup_latencies_ms": cache_lookup_latencies_ms,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "fallback_token_count": fallback_token_count,
    }


# ---------------------------------------------------------------------------
# 6. Metrics, Cost & Reporting
# ---------------------------------------------------------------------------


def evaluate_predictions(y_true: list[str], y_pred: list[str | None]) -> dict[str, float] | None:
    if len(y_true) != len(y_pred) or any(p not in {"positive", "negative"} for p in y_pred):
        return None
    valid_pred = [str(p) for p in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true, valid_pred)),
        "precision": float(precision_score(y_true, valid_pred, pos_label="positive", zero_division=0)),
        "recall": float(recall_score(y_true, valid_pred, pos_label="positive", zero_division=0)),
        "f1": float(f1_score(y_true, valid_pred, pos_label="positive", zero_division=0)),
    }


def estimate_cloud_cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float | None, str]:
    env_in = os.getenv("CLOUD_INPUT_PRICE_PER_1M", "").strip()
    env_out = os.getenv("CLOUD_OUTPUT_PRICE_PER_1M", "").strip()

    if env_in and env_out:
        in_rate, out_rate = float(env_in), float(env_out)
        source = "env override"
    elif model_name in MODEL_PRICING_PER_1M:
        in_rate, out_rate = MODEL_PRICING_PER_1M[model_name]
        source = "documented paid list price"
    else:
        return None, "unknown pricing (set CLOUD_INPUT_PRICE_PER_1M & CLOUD_OUTPUT_PRICE_PER_1M)"

    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return cost, f"{source} (${in_rate}/1M in, ${out_rate}/1M out)"


def format_latency_cell(run_stats: dict[str, Any]) -> str:
    live = run_stats["live_latencies_ms"]
    stored = run_stats["stored_latencies_ms"]
    if live:
        avg_ms = sum(live) / len(live)
        return f"{avg_ms:.2f}"
    if stored and run_stats["cache_hits"] == len(stored):
        avg_ms = sum(stored) / len(stored)
        return f"{avg_ms:.2f} (from prior live run; 100% cached)"
    return "Unavailable"


def build_shipping_analysis(rows_data: list[dict[str, Any]]) -> str:
    complete = [r for r in rows_data if r["metrics"] is not None]
    if len(complete) < 3:
        return (
            "_Experiment incomplete: run all three models cleanly on all 100 test reviews "
            "before finalizing the shipping decision._"
        )

    sklearn_row, cloud_row, ollama_row = rows_data
    best_by_f1 = max(complete, key=lambda r: r["metrics"]["f1"])
    sk_f1 = sklearn_row["metrics"]["f1"]
    best_f1 = best_by_f1["metrics"]["f1"]
    f1_gap = best_f1 - sk_f1

    if f1_gap <= 0.02:
        choice = sklearn_row["label"]
        rationale = (
            f"For this 500-review sentiment classification task, I would ship **{choice}**. "
            f"It achieved an F1-score of {sk_f1:.4f} and accuracy of {sklearn_row['metrics']['accuracy']:.4f}, "
            f"matching or staying within {abs(f1_gap):.4f} F1 of the best-performing LLM ({best_by_f1['label']} at {best_f1:.4f} F1) "
            f"while costing $0.00 in API fees and predicting in {sklearn_row['time_str']} ms per review. "
            f"The Cloud LLM ({cloud_row['label']}) adds network latency ({cloud_row['time_str']} ms/review), external API dependencies, "
            f"and paid-tier cost ({cloud_row['cost_str']}), while Ollama ({ollama_row['label']}) requires local GPU/CPU overhead "
            f"({ollama_row['time_str']} ms/review) without a meaningful accuracy gain. "
            "On well-structured, narrow binary text classification like product reviews, TF-IDF + Logistic Regression delivers the best balance of reliability, speed, and zero marginal inference cost."
        )
    else:
        rationale = (
            f"Based on the measured results, **{best_by_f1['label']}** achieved the highest F1-score ({best_f1:.4f}), "
            f"outperforming TF-IDF + Logistic Regression ({sk_f1:.4f} F1) by {f1_gap:.4f}. "
            f"Whether to ship {best_by_f1['label']} or {sklearn_row['label']} depends on your latency and budget constraints: "
            f"scikit-learn runs in {sklearn_row['time_str']} ms/review at $0.00 API cost, whereas {cloud_row['label']} takes {cloud_row['time_str']} ms/review "
            f"at {cloud_row['cost_str']} (paid list price) and {ollama_row['label']} takes {ollama_row['time_str']} ms/review on local hardware. "
            f"If the +{f1_gap:.4f} F1 improvement justifies higher latency and infrastructure complexity, ship {best_by_f1['label']}; otherwise, ship scikit-learn for high-throughput simplicity."
        )
    return rationale


def write_results_markdown(
    dataset_stats: dict[str, Any],
    rows_data: list[dict[str, Any]],
    notes: list[str],
    output_path: Path = RESULTS_PATH,
) -> None:
    lines = [
        "# Model Race — Experimental Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows_data:
        m = row["metrics"]
        if m is None:
            lines.append(
                f"| {row['label']} | Incomplete | Incomplete | Incomplete | Incomplete | {row['cost_str']} | {row['time_str']} |"
            )
        else:
            lines.append(
                f"| {row['label']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} | {row['cost_str']} | {row['time_str']} |"
            )

    if notes:
        lines.append("")
        lines.append("### Run Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")

    shipping_text = build_shipping_analysis(rows_data)
    lines.extend(
        [
            "",
            "### Which Model Would I Ship?",
            "",
            shipping_text,
            "",
            "### Class Balance and Accuracy",
            "",
            f"- **Positive reviews:** {dataset_stats['positive_count']} ({dataset_stats['positive_pct']:.2f}%)",
            f"- **Negative reviews:** {dataset_stats['negative_count']} ({dataset_stats['negative_pct']:.2f}%)",
            f"- **Dataset balance:** The dataset of {dataset_stats['total']} reviews is **{dataset_stats['balance_label']}**.",
            "",
            "Accuracy can be misleading on imbalanced datasets because a trivial classifier that always predicts the majority class can score high accuracy without learning useful patterns (for example, predicting only positive on a 90/10 dataset gives 90% accuracy despite 0% recall on negative reviews). "
            f"In this dataset, the split is {dataset_stats['positive_pct']:.2f}% positive vs. {dataset_stats['negative_pct']:.2f}% negative ({dataset_stats['balance_label']}), so accuracy aligns closely with F1-score here—yet reporting precision, recall, and F1 alongside accuracy remains essential to verify performance on both classes.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------------------------


async def main() -> None:
    df, dataset_stats = load_and_validate_dataset(
        CSV_PATH, TEXT_COLUMN, LABEL_COLUMN, expected_rows=500
    )

    X_train, X_test, y_train, y_test = split_dataset(df, TEXT_COLUMN, LABEL_COLUMN)
    print(
        f"\nSplit complete -> Train: {len(X_train)} reviews | Test: {len(X_test)} reviews "
        f"(random_state={RANDOM_STATE}, stratified)"
    )

    # Model 1: Traditional ML
    print("\n=== Running Model 1: scikit-learn (TF-IDF + Logistic Regression) ===")
    ml_preds, ml_ms_per_review, _ = run_traditional_ml(X_train, X_test, y_train)
    ml_metrics = evaluate_predictions(y_test, ml_preds)
    print(
        f"Done in {ml_ms_per_review:.4f} ms/review | "
        f"Accuracy={ml_metrics['accuracy']:.4f}, F1={ml_metrics['f1']:.4f}"
    )

    cache = load_cache(CACHE_PATH)

    # Model 2: Cloud LLM (Groq)
    print(f"\n=== Running Model 2: Cloud LLM (groq / {GROQ_MODEL}) ===")
    cloud_run = await run_llm_classification(
        reviews=X_test,
        provider="groq",
        model_name=GROQ_MODEL,
        cache=cache,
        cache_path=CACHE_PATH,
        request_delay=CLOUD_REQUEST_DELAY,
    )
    cloud_metrics = evaluate_predictions(y_test, cloud_run["predictions"])
    cloud_cost, pricing_note = estimate_cloud_cost_usd(
        GROQ_MODEL, cloud_run["input_tokens"], cloud_run["output_tokens"]
    )
    cloud_cost_str = f"${cloud_cost:.6f}" if cloud_cost is not None else "Unknown"
    cloud_time_str = format_latency_cell(cloud_run)

    print(
        f"Cloud LLM stats -> Live API calls: {cloud_run['live_calls']} | "
        f"Cache hits: {cloud_run['cache_hits']} | Failures: {len(cloud_run['failures'])}"
    )
    if cloud_run["cache_lookup_latencies_ms"]:
        avg_lookup = sum(cloud_run["cache_lookup_latencies_ms"]) / len(
            cloud_run["cache_lookup_latencies_ms"]
        )
        print(f"  Cache lookup speed (reported separately): {avg_lookup:.4f} ms/review")
    print(
        f"  Tokens -> input: {cloud_run['input_tokens']}, output: {cloud_run['output_tokens']} | "
        f"Out-of-pocket (free tier): $0.00 | Estimated paid list price: {cloud_cost_str} ({pricing_note})"
    )

    # Model 3: Local LLM (Ollama)
    print(f"\n=== Running Model 3: Local LLM (Ollama / {OLLAMA_MODEL}) ===")
    ollama_run = await run_llm_classification(
        reviews=X_test,
        provider="ollama",
        model_name=OLLAMA_MODEL,
        cache=cache,
        cache_path=CACHE_PATH,
        request_delay=0.0,
    )
    ollama_metrics = evaluate_predictions(y_test, ollama_run["predictions"])
    ollama_time_str = format_latency_cell(ollama_run)

    print(
        f"Ollama stats -> Live calls: {ollama_run['live_calls']} | "
        f"Cache hits: {ollama_run['cache_hits']} | Failures: {len(ollama_run['failures'])}"
    )
    if ollama_run["cache_lookup_latencies_ms"]:
        avg_lookup = sum(ollama_run["cache_lookup_latencies_ms"]) / len(
            ollama_run["cache_lookup_latencies_ms"]
        )
        print(f"  Cache lookup speed (reported separately): {avg_lookup:.4f} ms/review")

    rows_data = [
        {
            "label": "scikit-learn (TF-IDF + Logistic Regression)",
            "metrics": ml_metrics,
            "cost_str": "$0.000000",
            "time_str": f"{ml_ms_per_review:.4f}",
        },
        {
            "label": f"Cloud LLM (groq: {GROQ_MODEL})",
            "metrics": cloud_metrics,
            "cost_str": cloud_cost_str,
            "time_str": cloud_time_str,
        },
        {
            "label": f"Ollama ({OLLAMA_MODEL})",
            "metrics": ollama_metrics,
            "cost_str": "$0.000000 (local hardware & electricity not free)",
            "time_str": ollama_time_str,
        },
    ]

    notes = [
        f"Test set: {len(X_test)} stratified reviews (random_state={RANDOM_STATE}).",
        f"Cloud LLM ({GROQ_MODEL}) usage: {cloud_run['input_tokens']} input tokens, {cloud_run['output_tokens']} output tokens; out-of-pocket free-tier cost = $0.00; paid list estimate uses {pricing_note}.",
        f"Cloud LLM API calls made this run: {cloud_run['live_calls']} | Served from cache: {cloud_run['cache_hits']} | Failures: {len(cloud_run['failures'])}.",
        f"Ollama ({OLLAMA_MODEL}) live calls this run: {ollama_run['live_calls']} | Served from cache: {ollama_run['cache_hits']} | Failures: {len(ollama_run['failures'])}.",
    ]
    if cloud_run["fallback_token_count"] > 0 or ollama_run["fallback_token_count"] > 0:
        notes.append(
            "Where token metadata was unavailable, token counts were estimated via ceil(chars / 4)."
        )

    write_results_markdown(dataset_stats, rows_data, notes, RESULTS_PATH)
    print(f"\nResults written to '{RESULTS_PATH}'.")


if __name__ == "__main__":
    asyncio.run(main())