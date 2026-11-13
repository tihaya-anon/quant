"""数据层：从 baostock 下载日线数据、本地缓存、整理成宽表面板。

用法（在项目根目录）：
    uv run python -m quant.data              # 下载（已缓存的跳过）并生成面板
    uv run python -m quant.data --force      # 强制重新下载
    uv run python -m quant.data --build-only # 只用已缓存数据重建面板
"""

from __future__ import annotations

import argparse
import atexit
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import baostock as bs
import pandas as pd

from quant import config

STOCK_FIELDS = "date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,isST"
INDEX_FIELDS = "date,code,open,high,low,close,volume,amount"
NUMERIC = ["open", "high", "low", "close", "volume", "amount", "turn", "tradestatus", "peTTM", "pbMRQ", "isST"]

PANEL_FILE = config.PROCESSED_DIR / "panel.pkl"
UNIVERSE_FILE = config.PROCESSED_DIR / "universe.csv"


# ---------------------------------------------------------------- baostock 访问


def _login() -> None:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    atexit.register(bs.logout)


def _rows(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def fetch_universe(name: str = config.UNIVERSE) -> pd.DataFrame:
    """查询指数成分股（需已登录）。返回列: code, code_name。"""
    query = {"sz50": bs.query_sz50_stocks, "hs300": bs.query_hs300_stocks}[name]
    return _rows(query())[["code", "code_name"]]


def fetch_daily(code: str, is_index: bool = False) -> pd.DataFrame:
    """下载单只证券日线。股票用后复权价格（adjustflag=1），保证收益率计算连续。"""
    rs = bs.query_history_k_data_plus(
        code,
        INDEX_FIELDS if is_index else STOCK_FIELDS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        frequency="d",
        adjustflag="3" if is_index else "1",
    )
    df = _rows(rs)
    df["date"] = pd.to_datetime(df["date"])
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _download_one(code: str, is_index: bool, force: bool) -> tuple[str, int, str]:
    """子进程任务：下载并缓存单只证券。返回 (code, 行数, 状态)。"""
    path = config.RAW_DIR / f"{code}.pkl"
    if path.exists() and not force:
        return code, len(pd.read_pickle(path)), "cached"
    for attempt in range(3):
        try:
            df = fetch_daily(code, is_index)
            df.to_pickle(path)
            return code, len(df), "ok"
        except Exception as e:  # 网络抖动时重试
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return code, 0, f"failed: {err}"


# ---------------------------------------------------------------- 下载入口


def download(force: bool = False, workers: int = config.DOWNLOAD_WORKERS) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    _login()
    universe = fetch_universe()
    universe.to_csv(UNIVERSE_FILE, index=False)
    print(f"股票池 {config.UNIVERSE}: {len(universe)} 只")

    tasks = [(c, False) for c in universe["code"]] + [(c, True) for c in config.INDICES]
    done = 0
    # baostock 是单 socket 的全局会话，用多进程（每进程独立登录）并行
    with ProcessPoolExecutor(max_workers=workers, initializer=_login) as pool:
        futures = [pool.submit(_download_one, c, idx, force) for c, idx in tasks]
        for fut in as_completed(futures):
            code, n, status = fut.result()
            done += 1
            print(f"[{done}/{len(tasks)}] {code}: {n} 行 {status}", flush=True)


# ---------------------------------------------------------------- 面板构建


def build_panel() -> dict[str, pd.DataFrame]:
    """把每只证券的长表拼成宽表（index=日期, columns=代码）并保存。"""
    universe = pd.read_csv(UNIVERSE_FILE)
    stocks = pd.concat(
        [pd.read_pickle(config.RAW_DIR / f"{c}.pkl") for c in universe["code"]
         if (config.RAW_DIR / f"{c}.pkl").exists()],
        ignore_index=True,
    )
    indices = pd.concat([pd.read_pickle(config.RAW_DIR / f"{c}.pkl") for c in config.INDICES], ignore_index=True)

    # 以基准指数的交易日为统一日历
    calendar = pd.DatetimeIndex(sorted(indices.loc[indices["code"] == config.BENCHMARK, "date"].unique()))

    def wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
        return df.pivot(index="date", columns="code", values=field).reindex(calendar)

    panel = {f: wide(stocks, f) for f in ["open", "high", "low", "close", "volume", "amount", "turn", "peTTM", "pbMRQ"]}
    # 可交易：有行情、未停牌、非 ST
    status = wide(stocks, "tradestatus")
    is_st = wide(stocks, "isST")
    panel["tradable"] = (status == 1) & (is_st == 0) & panel["open"].notna()
    # 停牌日价格前值填充，用于估值（上市前保持 NaN）
    panel["close"] = panel["close"].ffill()
    panel["index_close"] = wide(indices, "close")
    panel["index_open"] = wide(indices, "open")
    panel["names"] = universe.set_index("code")["code_name"]

    pd.to_pickle(panel, PANEL_FILE)
    print(f"面板已保存: {PANEL_FILE}  ({len(calendar)} 个交易日 x {panel['close'].shape[1]} 只股票)")
    return panel


def load_panel() -> dict[str, pd.DataFrame]:
    if not PANEL_FILE.exists():
        raise FileNotFoundError("未找到面板数据，请先运行: uv run python -m quant.data")
    return pd.read_pickle(PANEL_FILE)


def main() -> None:
    p = argparse.ArgumentParser(description="下载 baostock 数据并构建面板")
    p.add_argument("--force", action="store_true", help="忽略缓存重新下载")
    p.add_argument("--build-only", action="store_true", help="只重建面板，不下载")
    p.add_argument("--workers", type=int, default=config.DOWNLOAD_WORKERS)
    args = p.parse_args()
    if not args.build_only:
        download(force=args.force, workers=args.workers)
    build_panel()


if __name__ == "__main__":
    main()
