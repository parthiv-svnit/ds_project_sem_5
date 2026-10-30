"""Tests the collector's logic against a MOCK of the openml package (no network).

These tests do NOT verify the live OpenML API; they verify error handling, fallbacks and the
shape of the raw records. The live run must be performed by the user (see README).
"""
import types

import pandas as pd

import collect_openml_metadata as col


class FakeDS:
    def __init__(self, did, fail_times=0):
        self.name, self.description, self.tag = f"ds{did}", "A description", ["a", "b"]
        self.default_target_attribute, self.licence, self.url = "y", "Public", f"https://x/{did}"


def make_fake_openml(fail_ids=(), output_format_supported=True):
    calls = {"get": []}
    def list_datasets(status=None, size=None, output_format=None):
        if output_format is not None and not output_format_supported:
            raise TypeError("unexpected keyword argument 'output_format'")
        df = pd.DataFrame({"did": [3, 1, 2, 2], "name": list("cabb"), "NumberOfInstances": [10, 20, 30, 30]})
        return df if output_format_supported else df.to_dict(orient="list") and df
    def get_dataset(did, download_data=True, download_qualities=True, download_features_meta_data=True):
        assert download_data is False, "collector must never download the data file"
        calls["get"].append(did)
        if did in fail_ids:
            raise ConnectionError("boom")
        return FakeDS(did)
    mod = types.SimpleNamespace(datasets=types.SimpleNamespace(list_datasets=list_datasets, get_dataset=get_dataset), __version__="fake")
    return mod, calls


def test_listing_is_deduplicated_and_sorted():
    mod, _ = make_fake_openml()
    lst = col.list_candidates(mod, "active", 100)
    assert list(lst["did"]) == [1, 2, 3]


def test_listing_falls_back_when_output_format_is_unsupported():
    mod, _ = make_fake_openml(output_format_supported=False)
    lst = col.list_candidates(mod, "active", 100)
    assert len(lst) == 3


def test_fetch_one_never_downloads_data_and_tolerates_missing_attributes():
    mod, calls = make_fake_openml()
    rec, err = col.fetch_one(mod, 1, with_features=False, max_retries=1, pause=0)
    assert err is None and rec["dataset_id"] == 1 and rec["tags"] == ["a", "b"]
    assert rec["row_id_attribute"] is None and rec["creator"] is None  # absent attributes -> None, no crash


def test_fetch_one_reports_error_after_retries(monkeypatch):
    monkeypatch.setattr(col.time, "sleep", lambda s: None)
    mod, calls = make_fake_openml(fail_ids={2})
    rec, err = col.fetch_one(mod, 2, with_features=False, max_retries=3, pause=0)
    assert rec is None and "ConnectionError" in err and calls["get"].count(2) == 3
