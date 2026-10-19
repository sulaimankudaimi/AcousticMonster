#!/usr/bin/env python3
"""Value-aware secret scanner.

Flags *leaked credential values*, not the mere presence of words such as
``secret`` (which is unavoidable: this project reads ``SOVEREIGN_VAULT_KEY_PATH``
and ``BRIDGE_IP_SECRET`` from the environment everywhere).

Two rules:

* HIGH  -- unambiguous credential formats (Fernet keys, PEM private keys, AWS
  access-key ids, GitHub/Slack/OpenAI tokens).
* ASSIGN -- a secret-like key assigned a string *literal* value, unless the value
  looks like a placeholder (``unit-test-*``, ``example``, ``${...}``, ...).

Usage::

    python scripts/scan_secrets.py                 # scan git-tracked files
    python scripts/scan_secrets.py path [path...]  # scan explicit paths/dirs
"""

import os
import re
import sys

HIGH = re.compile(
    r"[A-Za-z0-9_\-]{43}=|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}"
    r"|gh[pous]_[A-Za-z0-9]{20,}|xox[baprs]-|sk-[A-Za-z0-9]{20,}"
)
ASSIGN = re.compile(
    r"""(?i)\b(password|passwd|pwd|api[_-]?key|apikey|secret|token|access[_-]?key)"""
    r"""\b\s*[:=]\s*['"]([^'"]{6,})['"]"""
)
PLACEHOLDER = (
    "unit-test", "some-other", "example", "changeme", "placeholder", "your-",
    "os.environ", "getenv", "dummy", "test-", "redacted", "${", "<", "...",
)
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini",
            ".toml", ".sh", ".cff", ".example", ".csv", ".log", ""}


def scan_text(text):
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if HIGH.search(line):
            hits.append((i, "HIGH", line.strip()[:160]))
            continue
        m = ASSIGN.search(line)
        if m and not any(p in m.group(2).lower() for p in PLACEHOLDER):
            hits.append((i, "ASSIGN", line.strip()[:160]))
    return hits


def scan_paths(paths):
    findings = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    findings += _scan_file(os.path.join(root, fn))
        else:
            findings += _scan_file(p)
    return findings


def _scan_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in TEXT_EXT:
        return []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return []
    return [(f"{path}:{i}", kind, snippet) for i, kind, snippet in scan_text(text)]


def tracked_files():
    import shutil
    import subprocess

    mingit = r"C:\Users\00D2~1\AppData\Local\Temp\opencode\mingit\cmd\git.exe"
    exe = shutil.which("git") or (mingit if os.path.exists(mingit) else None)
    if not exe:
        return None
    out = subprocess.run([exe, "ls-files"], capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else None


def main(argv):
    paths = argv or tracked_files()
    if not paths:
        print("no paths and no git executable", file=sys.stderr)
        return 2
    findings = scan_paths(paths)
    for loc, kind, snippet in findings:
        print(f"{kind:7s} {loc}: {snippet}")
    print(f"scanned {len(paths)} path(s); findings: {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
