#!/usr/bin/env python3
"""Check the actual Git index or outbound history before sharing this project.

This conservative source-only policy complements .gitignore: even a forced add
of a dataset, output, notebook, or binary is rejected. Content checks are a
backstop, not a substitute for reviewing source before publication.
"""

import argparse
from functools import lru_cache
from pathlib import PurePosixPath
import re
import subprocess
import sys

IDENTITY = ("NDFitter contributors", "contributors@example.invalid")
ROOT_FILES = {".gitignore", ".gitattributes", "LICENSE", "README.md", "pyproject.toml"}
HOOK_FILES = {".githooks/pre-commit", ".githooks/pre-push"}
PRIVATE_DIRS = {"local", "outputs", "results", "runs", "__pycache__", ".git", ".venv", ".vscode"}
HOME_PATH = re.compile(r"/(?:Users|home)/[^/\s]+|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.I)
EMAIL = re.compile(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")


def git(*args):
    return subprocess.check_output(["git", *args])


def allowed_path(name):
    path = PurePosixPath(name)
    if any(part in PRIVATE_DIRS for part in path.parts):
        return False
    if "data" in path.parts and not (path.parent == PurePosixPath("NDFitter/MLP/data") and path.suffix == ".py"):
        return False
    if name in ROOT_FILES or name in HOOK_FILES:
        return True
    if path.parts[0] in {"NDFitter", "tests"}:
        return path.suffix == ".py"
    if path.parts[0] == "scripts":
        return path.suffix in {".py", ".sh"}
    return path.parts[0] == "docs" and (path.suffix == ".md" or name == "docs/migration.json")


def content_issue(content):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return "binary content"
    if "\0" in text:
        return "binary content"
    if HOME_PATH.search(text):
        return "personal home-directory path"
    if any(address != IDENTITY[1] for address in EMAIL.findall(text)):
        return "email address"
    if TOKEN.search(text):
        return "credential-like content"
    return None


@lru_cache(maxsize=None)
def blob_issue(oid):
    return content_issue(git("cat-file", "blob", oid))


def check_entries(entries):
    problems = []
    for name, mode, oid in entries:
        if not allowed_path(name):
            problems.append(f"{name}: not permitted by source-only policy")
        elif mode not in {"100644", "100755"}:
            problems.append(f"{name}: symlinks, submodules, and unresolved entries are not permitted")
        else:
            issue = blob_issue(oid)
            if issue:
                problems.append(f"{name}: {issue}")
    return problems


def index_entries():
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if record:
            info, name = record.split(b"\t", 1)
            mode, oid, stage = info.decode().split()
            yield name.decode(), mode if stage == "0" else "unmerged", oid


def tree_entries(commit):
    for record in git("ls-tree", "-r", "-z", commit).split(b"\0"):
        if record:
            info, name = record.split(b"\t", 1)
            mode, _, oid = info.decode().split()
            yield name.decode(), mode, oid


def check_history(tips):
    problems = []
    seen = set()
    for tip in tips:
        if git("cat-file", "-t", tip).strip() != b"commit":
            problems.append("Only commit tips may be shared; annotated tags need a separate privacy review")
            continue
        for commit in git("rev-list", tip).decode().splitlines():
            if commit in seen:
                continue
            seen.add(commit)
            identity = git("show", "-s", "--format=%an%n%ae%n%cn%n%ce", commit).decode().splitlines()
            if tuple(identity) != IDENTITY + IDENTITY:
                problems.append(f"{commit[:12]}: author/committer identity is not anonymized")
            message = git("show", "-s", "--format=%B", commit)
            if content_issue(message):
                problems.append(f"{commit[:12]}: private content in commit message")
            problems.extend(f"{commit[:12]}: {issue}" for issue in check_entries(tree_entries(commit)))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--history", action="store_true", help="Check all commits reachable from HEAD")
    group.add_argument("--pre-push", action="store_true", help="Check every outbound ref received on stdin")
    args = parser.parse_args()
    if args.pre_push:
        tips = []
        for line in sys.stdin:
            _, local_oid, _, _ = line.split()
            if set(local_oid) != {"0"}:  # Deleting a remote ref sends no content.
                tips.append(local_oid)
        problems = check_history(tips)
    elif args.history:
        problems = check_history(["HEAD"])
    else:
        problems = check_entries(index_entries())
    if problems:
        print("Repository privacy check failed:", file=sys.stderr)
        print("\n".join(problems[:30]), file=sys.stderr)
        if len(problems) > 30:
            print(f"... plus {len(problems) - 30} more issues", file=sys.stderr)
        return 1
    print("Repository privacy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
