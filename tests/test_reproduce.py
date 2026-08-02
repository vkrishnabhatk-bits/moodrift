"""Guards on the checks that stand between "re-ran it" and "reproduced it".

The retrain itself is exercised for real by `make reproduce RUN_ID=<id>`; what is unit
tested here is the part that decides whether a re-run is even comparable - config drift and
commit drift. Both fail silently if they are wrong: the training script would happily use
today's config and today's code and report a number that matches by luck.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.train import reproduce


def _run(params: dict[str, str], tags: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(params=params, tags=tags or {}, metrics={}))


class TestFlatten:
    def test_nested_config_flattens_to_dotted_keys(self):
        flat = reproduce._flatten({"a": 1, "b": {"c": "x", "d": {"e": 2}}})
        assert flat == {"a": "1", "b.c": "x", "b.d.e": "2"}

    def test_values_are_stringified_like_mlflow_logs_them(self):
        # MLflow stores params as strings; comparing raw Python values against them would
        # report drift on every list and every float.
        assert reproduce._flatten({"ngram_range": [1, 2]}) == {"ngram_range": "[1, 2]"}


class TestCheckConfig:
    def test_matching_config_reports_no_problems(self, monkeypatch):
        monkeypatch.setattr(reproduce, "load_config", lambda name: {"seed": 42, "nested": {"a": 1}})
        run = _run({"seed": "42", "nested.a": "1"})
        assert reproduce.check_config("1", run) == []

    def test_changed_value_is_reported(self, monkeypatch):
        monkeypatch.setattr(reproduce, "load_config", lambda name: {"seed": 7})
        problems = reproduce.check_config("1", _run({"seed": "42"}))
        assert len(problems) == 1 and "seed" in problems[0]

    def test_new_config_key_is_reported(self, monkeypatch):
        monkeypatch.setattr(reproduce, "load_config", lambda name: {"seed": 42, "added": True})
        problems = reproduce.check_config("1", _run({"seed": "42"}))
        assert len(problems) == 1 and "added" in problems[0]


class TestCheckCode:
    def test_same_commit_is_silent(self, monkeypatch):
        monkeypatch.setattr(reproduce, "git_sha", lambda: "abc123")
        assert reproduce.check_code("abc123") == []

    def test_different_commit_warns(self, monkeypatch):
        monkeypatch.setattr(reproduce, "git_sha", lambda: "def456")
        assert "def456"[:8] in reproduce.check_code("abc123")[0]

    def test_dirty_tree_at_the_right_commit_still_warns(self, monkeypatch):
        # The commit matches but the files on disk do not, which is the sneakier case:
        # everything looks right and the code that ran is not the code that is committed.
        monkeypatch.setattr(reproduce, "git_sha", lambda: "abc123-dirty")
        warnings = reproduce.check_code("abc123")
        assert len(warnings) == 1 and "dirty" in warnings[0]
