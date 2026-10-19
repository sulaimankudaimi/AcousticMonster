#!/usr/bin/env python3
"""Fetch and verify the GPLA-12 acoustic dataset used by E5.

The dataset is pinned to a single commit and every file is checked against the
sha256 recorded in ``data/SOURCE.md`` -- nothing is trusted from the network.

Usage::

    python scripts/fetch_gpla12.py                 # download at the pinned commit
    python scripts/fetch_gpla12.py --archive X.zip # verify a local zip (offline)
    python scripts/fetch_gpla12.py --check         # verify existing files only

Exit status is non-zero if any file is missing or its sha256 does not match.

Dataset: J. Li, L. Yao, "GPLA-12: An Acoustic Signal Dataset of Gas Pipeline
Leakage", arXiv:2106.10277. Repository license: EPL-2.0 (see data/SOURCE.md).
"""

import argparse
import hashlib
import os
import re
import sys
import zipfile

COMMIT = "afec271f4eb445a976826c6a45485892e5fbab53"
REPO = "Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12"
ARCHIVE_URL = f"https://codeload.github.com/{REPO}/zip/{COMMIT}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "SOURCE.md")
DEST = os.path.join(ROOT, "data", "external", "gpla12")


def expected_hashes():
    """Parse ``| path | size | sha256 |`` rows from data/SOURCE.md."""
    exp = {}
    row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([0-9A-Fa-f]{64})\s*")
    with open(SOURCE, encoding="utf-8") as fh:
        for line in fh:
            m = row.match(line)
            if m:
                exp[m.group(1).strip()] = m.group(3).upper()
    return exp


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def download(dest_zip):
    import urllib.request

    print(f"downloading {ARCHIVE_URL}")
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as resp, open(dest_zip, "wb") as out:
        out.write(resp.read())


def extract(zip_path):
    """Extract the repo files (stripping the top-level commit folder)."""
    os.makedirs(DEST, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        top = zf.namelist()[0].split("/")[0] + "/"
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            rel = name[len(top):] if name.startswith(top) else name
            if not rel:
                continue
            out = os.path.join(DEST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())


def verify(expected):
    ok = True
    for rel, want in expected.items():
        path = os.path.join(DEST, rel)
        if not os.path.exists(path):
            print(f"MISSING  {rel}")
            ok = False
        else:
            got = sha256(path)
            status = "OK" if got == want else "MISMATCH"
            if got != want:
                ok = False
            print(f"{status:8s} {rel}  {got}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="local zip instead of downloading")
    ap.add_argument("--check", action="store_true", help="verify existing files only")
    args = ap.parse_args()

    expected = expected_hashes()
    if not expected:
        print("no sha256 rows parsed from data/SOURCE.md", file=sys.stderr)
        return 2

    if not args.check:
        zip_path = args.archive
        tmp = None
        if not zip_path:
            tmp = os.path.join(DEST, "_gpla12.zip")
            os.makedirs(DEST, exist_ok=True)
            download(tmp)
            zip_path = tmp
        print(f"extracting {zip_path} -> {DEST}")
        extract(zip_path)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    ok = verify(expected)
    print("gpla12 verification:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
