#!/usr/bin/env bash
# 数据通过私有仓库的 GitHub Release 分发（含购买的数据包，不能放进公开的代码仓库）。
#
#   scripts/data_release.sh upload [tag]   # 打包 data/ 并上传到 $DATA_REPO 的 Release（默认 tag: data-YYYYMMDD）
#   scripts/data_release.sh fetch  [tag]   # 下载并还原到 data/（默认取最新 Release）
#
# 资产：
#   local-<年>.parquet / local-adj_factor.parquet / local-names.parquet  -> data/local/...
#   processed.tar.gz  面板与成分（data/processed）
#   raw.tar.gz        baostock 下载缓存（data/raw）
#   results.tar.gz    回测结果（data/results）
# 需要 gh 已登录且有 $DATA_REPO 的读权限。
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_REPO=${DATA_REPO:-tihaya-anon/quant-data}
cmd=${1:-}
tag=${2:-}

case "$cmd" in
  upload)
    tag=${tag:-data-$(date +%Y%m%d)}
    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    for f in data/local/daily/*.parquet; do ln -s "$PWD/$f" "$stage/local-$(basename "$f")"; done
    for f in data/local/*.parquet; do ln -s "$PWD/$f" "$stage/local-$(basename "$f")"; done
    for d in processed raw results; do
      echo "打包 data/$d ..."
      tar -czf "$stage/$d.tar.gz" -C data "$d"
    done
    (cd "$stage" && sha256sum *  > SHA256SUMS)
    notes="数据快照 $(date '+%F')。由 scripts/data_release.sh upload 生成，用 fetch 还原。"
    if gh release view "$tag" -R "$DATA_REPO" > /dev/null 2>&1; then
      gh release upload "$tag" -R "$DATA_REPO" --clobber "$stage"/*
    else
      gh release create "$tag" -R "$DATA_REPO" --title "$tag" --notes "$notes" "$stage"/*
    fi
    echo "已上传: https://github.com/$DATA_REPO/releases/tag/$tag"
    ;;
  fetch)
    dl=$(mktemp -d)
    trap 'rm -rf "$dl"' EXIT
    if [[ -n "$tag" ]]; then gh release download "$tag" -R "$DATA_REPO" -D "$dl"
    else gh release download -R "$DATA_REPO" -D "$dl"; fi
    (cd "$dl" && sha256sum -c SHA256SUMS --quiet)
    mkdir -p data/local/daily
    for f in "$dl"/local-*.parquet; do
      name=$(basename "$f"); name=${name#local-}
      if [[ $name =~ ^[0-9]{4}\.parquet$ ]]; then mv "$f" "data/local/daily/$name"; else mv "$f" "data/local/$name"; fi
    done
    for d in processed raw results; do tar -xzf "$dl/$d.tar.gz" -C data; done
    echo "已还原到 data/"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
