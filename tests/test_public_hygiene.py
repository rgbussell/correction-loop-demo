"""Public-readiness guard (G5): this repo is PUBLIC, so identifying or
proprietary strings must fail a test, not wait for a manual audit.

Two layers. The generic patterns below are safe to publish and always run:
home-directory paths, e-mail addresses, cloud-storage locations, credential
shapes. Anything that would itself be a disclosure — an employer's name, a
product, a project codename — lives in an UNTRACKED local deny-list
(``.hygiene-denylist.local``, one case-insensitive regex per line, ``#``
comments) or the ``CLLOOP_DENYLIST`` env var, so the guard never publishes
the thing it guards. Tracked file contents AND every commit message are
scanned; the pre-commit hook (scripts/install_hooks.sh) runs this file.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

GENERIC = {
    "home-directory path": r"(/home/|/Users/)[A-Za-z0-9_.-]+",
    "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[a-z]{2,}",
    "cloud-storage location": r"(gdrive|s3|gs|azure)://\S+|drive\.google\.com/\S+",
    "private key": r"BEGIN [A-Z ]*PRIVATE KEY",
    "credential assignment": r"(client_secret|api[_-]?key|access[_-]?token|password)"
                             r"[\"']?\s*[:=]\s*[\"'][^\"']{8,}",
}
SELF = "tests/test_public_hygiene.py"  # the patterns above would match themselves


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout


def _local_denylist() -> dict[str, str]:
    lines: list[str] = os.environ.get("CLLOOP_DENYLIST", "").split("|")
    p = REPO / ".hygiene-denylist.local"
    if p.is_file():
        lines += p.read_text().splitlines()
    terms = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return {f"local deny-list term #{i + 1}": t for i, t in enumerate(terms)}


def _scan(text: str, patterns: dict[str, str]) -> list[tuple[int, str]]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, pat in patterns.items():
            if re.search(pat, line, flags=re.IGNORECASE):
                hits.append((n, name))
    return hits


def _tracked_text_files():
    # staged-or-committed paths, so the hook sees a file the moment it is added
    for rel in sorted(set(_git("ls-files", "--cached").splitlines())):
        p = REPO / rel
        if rel == SELF or not p.is_file():
            continue
        try:
            yield rel, p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary — DVC pointers keep model bytes out anyway


def test_tracked_files_carry_nothing_identifying():
    patterns = {**GENERIC, **_local_denylist()}
    bad = [f"{rel}:{n}: {name}" for rel, text in _tracked_text_files()
           for n, name in _scan(text, patterns)]
    assert not bad, "identifying/proprietary content in tracked files:\n" + "\n".join(bad)


def test_commit_messages_carry_nothing_identifying():
    patterns = {**GENERIC, **_local_denylist()}
    bad = [f"commit message line {n}: {name}"
           for n, name in _scan(_git("log", "--all", "--format=%B"), patterns)]
    assert not bad, "\n".join(bad)


def test_the_guard_can_see_a_planted_violation():
    """The guard's own null: a scanner that cannot fire proves nothing."""
    planted = "PY=/home/someone/miniconda3/bin/python\ncontact: a.person@example.org\n"
    names = {name for _, name in _scan(planted, GENERIC)}
    assert {"home-directory path", "e-mail address"} <= names
    assert _scan("acme corp data", {"t": "acme"}) and not _scan("clean line", GENERIC)


def test_dvc_remote_is_never_tracked():
    assert _git("show", "HEAD:.dvc/config").strip() == "", \
        "a DVC remote is committed — it belongs in .dvc/config.local only"
    assert not _git("ls-files", ".dvc/config.local").strip()


@pytest.mark.skipif(not (REPO / ".hygiene-denylist.local").is_file()
                    and not os.environ.get("CLLOOP_DENYLIST"),
                    reason="no local deny-list on this machine (generic layer still ran)")
def test_local_denylist_is_itself_untracked():
    assert not _git("ls-files", ".hygiene-denylist.local").strip()
