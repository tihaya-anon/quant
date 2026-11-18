#!/usr/bin/env bash
# 全流程重跑（后台，关掉终端也不会中断）：
#   1. 面板：本地数据包 -> data/processed/panel_<股票池>.pkl          约 2 分钟
#   2. 向量化回测：四个基础版 + 四个增强版                               约 10 分钟
#   3. 执行仿真：backtrader 42 次回测，多进程                            约 1 小时
#   4. 学习笔记：quarto render report -> report/_site/                 约 1 分钟
#
# 用法：
#   scripts/rebuild_all.sh                 # 全部
#   scripts/rebuild_all.sh --workers=8     # 执行仿真的进程数（默认 6，每进程约 1GB 内存）
#   scripts/rebuild_all.sh --skip-exec     # 跳过执行仿真（只想快速看向量化结果时）
#
# 查看进度：  tail -f logs/rebuild_all.log
# 中途停止：  pkill -f rebuild_all.sh; pkill -f quant.run_exec
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/rebuild_all.log

if [[ "${1:-}" != "--run" ]]; then  # 首次调用：检查重复运行，然后把自己放到后台
  if pgrep -f "rebuild_all.sh --run" > /dev/null || pgrep -f "python.*-m quant.run_exec" > /dev/null; then
    echo "已有重跑任务在运行" >&2
    exit 1
  fi
  echo "=== $(date '+%F %T') 启动 rebuild_all $* (UNIVERSE=${QUANT_UNIVERSE:-默认}) ===" >> "$LOG"
  nohup "$0" --run "$@" >> "$LOG" 2>&1 &
  echo "已在后台启动 (PID $!)，日志: $LOG"
  echo "查看进度: tail -f $LOG"
  exit 0
fi
shift

SKIP_EXEC=0
EXEC_ARGS=()
for a in "$@"; do
  case "$a" in
    --skip-exec) SKIP_EXEC=1 ;;
    *) EXEC_ARGS+=("$a") ;;
  esac
done

export PYTHONWARNINGS=ignore
step() { echo; echo "=== $(date '+%F %T') $* ==="; }

step "1/4 构建面板"
uv run python -u -m quant.local_data build
step "2/4 向量化回测"
uv run python -u -m quant.run
if [[ $SKIP_EXEC == 0 ]]; then
  step "3/4 执行仿真"
  uv run python -u -m quant.run_exec "${EXEC_ARGS[@]}"
else
  step "3/4 执行仿真（跳过）"
fi
step "4/4 渲染学习笔记"
uv run quarto render report
step "完成：浏览器打开 report/_site/index.html，或 uv run quarto preview report"
