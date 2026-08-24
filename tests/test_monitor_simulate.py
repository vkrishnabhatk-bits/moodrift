import random

import pandas as pd
import pytest

from src.monitor import simulate


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return simulate._pool()


def test_pool_excludes_reference_rows(pool):
    ref = simulate._reference()
    assert set(ref["Id"]).isdisjoint(set(pool["Id"]))


@pytest.mark.parametrize("name", list(simulate.SCENARIOS))
def test_scenario_injection_preserves_row_count(pool, name):
    rows = pool.sample(n=20, random_state=1).reset_index(drop=True)
    rng = random.Random(0)
    out = simulate.SCENARIOS[name](rows, pool, rng)
    assert len(out) == len(rows)
    assert set(out.columns) >= {"Text", "Score"}


def test_vocabulary_shift_appends_slang_without_changing_score(pool):
    rows = pool.sample(n=10, random_state=1).reset_index(drop=True)
    rng = random.Random(0)
    out = simulate._inject_vocabulary_shift(rows, rng)
    assert (out["Score"] == rows["Score"]).all()
    assert all(out["Text"].str.len() > rows["Text"].str.len())


def test_topic_shift_uses_only_the_curated_electronics_pool(pool):
    rows = pool.sample(n=10, random_state=1).reset_index(drop=True)
    rng = random.Random(0)
    out = simulate._inject_topic_shift(rows, rng)
    valid_texts = {t for t, _ in simulate._ELECTRONICS_REVIEWS}
    assert set(out["Text"]).issubset(valid_texts)
    assert set(out["Score"]).issubset({s for _, s in simulate._ELECTRONICS_REVIEWS})


def test_length_shift_swaps_in_longer_real_text(pool):
    rows = pool.sample(n=10, random_state=1).reset_index(drop=True)
    rng = random.Random(0)
    out = simulate._inject_length_shift(rows, pool, rng)
    assert out["Text"].str.len().mean() > rows["Text"].str.len().mean()
    # every swapped-in text must be real - drawn from the pool, not fabricated
    assert set(out["Text"]).issubset(set(pool["Text"]))


def test_label_noise_contradicts_surface_sentiment_but_keeps_the_score(pool):
    rows = pool[pool["Score"] >= 4].sample(n=10, random_state=1).reset_index(drop=True)
    rng = random.Random(0)
    out = simulate._inject_label_noise(rows, pool, rng)
    assert (out["Score"] == rows["Score"]).all()  # label unchanged
    assert (out["Text"].to_numpy() != rows["Text"].to_numpy()).all()  # text swapped
    # swapped-in text should come from the low-rating pool for a high-scored row
    low_pool = set(pool[pool["Score"] <= 2]["Text"])
    assert set(out["Text"]).issubset(low_pool)


def test_build_window_at_zero_fraction_is_unmodified_pool_sample(pool):
    window = simulate._build_window(pool, "vocabulary_shift", fraction=0.0, seed=1)
    assert set(window["Text"]).issubset(set(pool["Text"]))


def test_build_window_respects_the_configured_size(pool):
    window = simulate._build_window(pool, "length_shift", fraction=0.3, seed=1)
    from src.config import load_config

    assert len(window) == int(load_config("monitor")["windows"]["size"])
