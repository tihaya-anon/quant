#!/usr/bin/env bash
# 后台运行执行仿真（backtrader，42 次回测，多进程并行），关掉终端也不会中断。
#
# 用法：
#   scripts/run_exec.sh                 # 默认 6 个进程
#   scripts/run_exec.sh --workers=8     # 指定进程数（机器 10 核，建议 ≤8；内存约每进程 1GB）
#   scripts/run_exec.sh --quick         # 只跑小资金（资金容量 3 档），更快
#   QUANT_UNIVERSE=cap800 scripts/run_exec.sh   # 换股票池（需先 build 对应面板并运行 quant.run）
#
# 查看进度：  tail -f logs/run_exec.log
# 中途停止：  pkill -f quant.run_exec
# 跑完后结果在 data/results/<股票池>/execution.pkl，再 uv run quarto render report 更新笔记。
# 全流程（面板 + 回测 + 执行 + 笔记）用 scripts/rebuild_all.sh。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/run_exec.log

if pgrep -f "python.*-m quant.run_exec" > /dev/null; then
  echo "已有 run_exec 在运行，先 pkill -f quant.run_exec 再启动" >&2
  exit 1
fi

echo "=== $(date '+%F %T') 启动 run_exec $* (UNIVERSE=${QUANT_UNIVERSE:-默认}) ===" >> "$LOG"
PYTHONWARNINGS=ignore nohup uv run python -u -m quant.run_exec "$@" >> "$LOG" 2>&1 &
echo "已在后台启动 (PID $!)，日志: $LOG"
echo "查看进度: tail -f $LOG"
