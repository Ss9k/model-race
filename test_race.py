from pathlib import Path

import pandas as pd
import pytest

from race import (
    build_cache_key,
    evaluate_predictions,
    load_and_validate_dataset,
    load_cache,
    normalize_sentiment,
    run_traditional_ml,
    save_cache,
    split_dataset,
)


def _sample_df(n_per_class: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append({"review": f"Great product quality number {i}", "sentiment": "positive"})
        rows.append({"review": f"Terrible broken item number {i}", "sentiment": "negative"})
    return pd.DataFrame(rows)


def test_split_reproducible_and_stratified():
    df = _sample_df(25)
    X1_tr, X1_te, y1_tr, y1_te = split_dataset(df, "review", "sentiment")
    X2_tr, X2_te, y2_tr, y2_te = split_dataset(df, "review", "sentiment")

    assert len(X1_tr) == 40
    assert len(X1_te) == 10
    assert X1_te == X2_te
    assert y1_te == y2_te
    assert y1_te.count("positive") == 5
    assert y1_te.count("negative") == 5


def test_tfidf_fit_only_on_train_no_leakage():
    X_train = ["great battery life", "terrible screen quality"] * 5
    X_test = ["unseenuniquetesttoken great battery", "unseenuniquetesttoken terrible screen"]
    y_train = ["positive", "negative"] * 5

    preds, ms_per_review, vectorizer = run_traditional_ml(X_train, X_test, y_train)
    assert len(preds) == 2
    assert ms_per_review >= 0
    assert "unseenuniquetesttoken" not in vectorizer.vocabulary_


def test_normalize_sentiment_strict():
    assert normalize_sentiment("positive") == "positive"
    assert normalize_sentiment("  NEGATIVE. ") == "negative"
    assert normalize_sentiment("positive because it works") is None
    assert normalize_sentiment("") is None


def test_validate_dataset_rejects_bad_labels(tmp_path: Path):
    csv_file = tmp_path / "bad.csv"
    pd.DataFrame(
        {"review": ["good", "okay"], "sentiment": ["positive", "neutral"]}
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="positive"):
        load_and_validate_dataset(csv_file, "review", "sentiment", expected_rows=None)


def test_cache_roundtrip_and_distinct_keys(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    k1 = build_cache_key("gemini", "gemini-3.5-flash-lite", "Great phone")
    k2 = build_cache_key("ollama", "llama3.2", "Great phone")
    assert k1 != k2

    save_cache({k1: {"prediction": "positive", "input_tokens": 10, "output_tokens": 1}}, cache_path)
    loaded = load_cache(cache_path)
    assert loaded[k1]["prediction"] == "positive"


def test_evaluate_predictions_rejects_incomplete():
    assert evaluate_predictions(["positive", "negative"], ["positive", None]) is None
    metrics = evaluate_predictions(["positive", "negative"], ["positive", "negative"])
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}