#!/usr/bin/env bash
# 生成 results/ 下全部 .dat 的 md5 清单 → data/MANIFEST.md5
# (results/ 不入 git;审阅者按 docs/REPRODUCING.md 重生成后与清单比对)
set -u; cd "$(dirname "$0")/.."
mkdir -p data
find results -name "*.dat" -print0 | sort -z | xargs -0 md5sum > data/MANIFEST.md5
echo "$(wc -l < data/MANIFEST.md5) files -> data/MANIFEST.md5"
