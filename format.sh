#!/usr/bin/env bash
set -euo pipefail

# ─── Configure these ──────────────────────────────────────────────────────────

DIRECTORIES=(
    include
    src
    tests
)

EXTENSIONS=(
    cpp
    hpp
    h
)

# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dir in "${DIRECTORIES[@]}"; do
    abs_dir="$REPO_ROOT/$dir"
    [[ -d "$abs_dir" ]] || continue
    for ext in "${EXTENSIONS[@]}"; do
        find "$abs_dir" -type f -name "*.$ext" -exec clang-format -i {} +
    done
done
