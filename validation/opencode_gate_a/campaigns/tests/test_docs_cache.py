"""Offline tests for the pinned documentation cache tooling (no network: the HTTP getter is injected)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "docs_cache", Path(__file__).resolve().parents[1] / "docs_cache.py"
)
assert _SPEC and _SPEC.loader
dc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dc)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _html(key: str, size: int = 12_000) -> str:
    body = (f"<p>{key} real documentation sentence about Lambda asynchronous invocation retries.</p>\n") * (size // 80)
    return f"<html><head><script>var x=1;</script></head><body>{body}</body></html>"


def _getter(url: str) -> str:
    return _html(url.rsplit("/", 1)[-1])


def test_fetch_pins_every_page_and_never_writes_a_stub(tmp_path: Path) -> None:
    manifest = dc.fetch(tmp_path, REPO_ROOT, getter=_getter)
    assert len(manifest["pages"]) == 12
    for key, entry in manifest["pages"].items():
        text = (tmp_path / f"{key}.txt").read_text(encoding="utf-8")
        assert not dc.is_stub(text)
        assert entry["sha256"] == dc.sha256_text(text)
        assert "<script" not in text
    assert dc.verify(tmp_path) == []


def test_lambda_invocation_is_the_declared_composition_of_the_async_pages(tmp_path: Path) -> None:
    fetched: list[str] = []

    def recording(url: str) -> str:
        fetched.append(url)
        return _html(url.rsplit("/", 1)[-1])

    manifest = dc.fetch(tmp_path, REPO_ROOT, getter=recording)
    entry = manifest["pages"]["lambda_invocation"]
    assert entry["sources"] == list(dc.COMPOSED_PAGES["lambda_invocation"])
    assert all(source in fetched for source in entry["sources"])
    text = (tmp_path / "lambda_invocation.txt").read_text(encoding="utf-8")
    # One block per source, in order; the other keys are exactly their corpus URL.
    assert text.count("real documentation sentence") >= 4 * (12_000 // 80) - 4
    assert manifest["pages"]["s3_naming"]["sources"] == [manifest["pages"]["s3_naming"]["url"]]


def test_fetch_fails_closed_on_a_short_or_stub_page(tmp_path: Path) -> None:
    def short(url: str) -> str:
        return "<p>Access denied</p>"

    with pytest.raises(RuntimeError, match="not a real page"):
        dc.fetch(tmp_path, REPO_ROOT, getter=short)


def test_fetch_fails_closed_on_a_transport_error(tmp_path: Path) -> None:
    def boom(url: str) -> str:
        raise OSError("network down")

    with pytest.raises(RuntimeError, match="could not download"):
        dc.fetch(tmp_path, REPO_ROOT, getter=boom)


def test_verify_detects_a_stub_a_changed_page_and_a_missing_page(tmp_path: Path) -> None:
    dc.fetch(tmp_path, REPO_ROOT, getter=_getter)
    (tmp_path / "lambda_invocation.txt").write_text("[offline stub for lambda_invocation] x\n" * 400, encoding="utf-8")
    (tmp_path / "s3_naming.txt").write_text(("different but long enough page text. " * 400), encoding="utf-8")
    (tmp_path / "iam_best_practices.txt").unlink()
    problems = dc.verify(tmp_path)
    assert any(p.startswith("lambda_invocation: stub") for p in problems)
    assert any(p.startswith("s3_naming: content differs") for p in problems)
    assert any(p.startswith("iam_best_practices: missing") for p in problems)


def test_verify_detects_a_tampered_manifest(tmp_path: Path) -> None:
    dc.fetch(tmp_path, REPO_ROOT, getter=_getter)
    path = tmp_path / dc.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["pages"]["s3_naming"]["sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("manifest digest" in p for p in dc.verify(tmp_path))


def test_install_replaces_only_stubs_and_refuses_a_different_real_page(tmp_path: Path) -> None:
    source, target = tmp_path / "staging", tmp_path / "cache"
    dc.fetch(source, REPO_ROOT, getter=_getter)
    target.mkdir()
    (target / "lambda_invocation.txt").write_text("[offline stub for lambda_invocation] x\n" * 400, encoding="utf-8")
    assert dc.install(source, target) == []
    assert dc.verify(target, source / dc.MANIFEST_NAME) == []

    (target / "s3_naming.txt").write_text("someone else's real page. " * 600, encoding="utf-8")
    problems = dc.install(source, target)
    assert any("refusing to overwrite" in p for p in problems)


def test_install_copies_nothing_when_the_source_does_not_verify(tmp_path: Path) -> None:
    source, target = tmp_path / "staging", tmp_path / "cache"
    dc.fetch(source, REPO_ROOT, getter=_getter)
    (source / "eks_networking.txt").write_text("short", encoding="utf-8")
    assert dc.install(source, target) != []
    assert not target.exists() or not any(target.iterdir())


def test_manifest_digest_ignores_fetch_time(tmp_path: Path) -> None:
    first = dc.fetch(tmp_path / "a", REPO_ROOT, getter=_getter)
    second = dc.fetch(tmp_path / "b", REPO_ROOT, getter=_getter)
    assert first["manifest_digest"] == second["manifest_digest"]


def test_cli_verify_exit_codes(tmp_path: Path) -> None:
    dc.fetch(tmp_path, REPO_ROOT, getter=_getter)
    assert dc.main(["docs_cache", "verify", "--dir", str(tmp_path)]) == 0
    (tmp_path / "rds_backups.txt").unlink()
    assert dc.main(["docs_cache", "verify", "--dir", str(tmp_path)]) == 1
