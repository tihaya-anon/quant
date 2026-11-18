"""本地数据源：购买的"全A日K + 复权因子"数据包 -> 项目统一面板。

两步（都可重复运行，已完成的部分自动跳过）：
    uv run python -m quant.local_data convert   # 读 D 盘 zip，按年转成 parquet 存到 data/local/（慢，只需一次）
    uv run python -m quant.local_data build     # 由 parquet 构建面板 data/processed/panel_<UNIVERSE>.pkl（快）

数据包格式（已核对）：
- 全A日K/<年>.zip -> <年>/<代码>.csv，列：code, datetime, OHLC(不复权), pre_close, volume(手), amount(千元),
  turnover(%), pe_ttm, pb, dv_ttm, total_mv/circ_mv(万元) 等
- 复权因子_后复权.zip -> <代码>.csv，列：股票代码, 交易日期, 复权因子；后复权价 = 不复权价 × 因子
- 注意：数据包**不含已退市股票**（只有复权因子包有），也没有 ST 标记和指数成分。

股票池（point-in-time，按当时的数据选）：
    cap300 / cap800：每月最后一个交易日，在已上市满 MIN_LISTED_DAYS 天的股票中，
    按**当时的**流通市值取前 N 名，从该日起生效到下一次调整。只用当时已知的数据，无未来函数。
    大市值股票极少退市，因此"数据包缺退市股"对这个股票池的影响较小（但不为零）。
"""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

from quant import config

SRC = Path(os.environ.get("QUANT_LOCAL_SRC", "/mnt/d/极速下载好的文件"))
DAILY_DIRS = [SRC / "我的资源" / "全A日K", SRC / "股票历史数据2000年至今" / "全A日K"]
ADJ_ZIP = SRC / "股票历史数据2000年至今" / "复权因子" / "复权因子_后复权.zip"

LOCAL_DIR = config.DATA_DIR / "local"
DAILY_OUT = LOCAL_DIR / "daily"  # <年>.parquet（长表，不复权原始字段）
ADJ_OUT = LOCAL_DIR / "adj_factor.parquet"
NAMES_OUT = LOCAL_DIR / "names.parquet"
# 名称来源：分钟数据包。注意每个文件里的名称都是卖家打包时的名称（不是历史名称），
# 所以只能用作"代码 -> 当前名称"的显示映射，不能用来推断历史 ST 状态（会引入未来信息）。
MINUTE_DIRS = [SRC / "股票历史数据2000年至今" / "沪深分钟数据" / "Stock_30min_2000-now",
               SRC / "股票历史数据2000年至今" / "京市分钟数据" / "StockJ_30min_2005-now"]

KEEP = ["code", "datetime", "open", "high", "low", "close", "pre_close", "volume", "amount", "turnover",
        "pe_ttm", "pb", "dv_ttm", "total_mv", "circ_mv"]
CAP_UNIVERSES = {"cap300": 300, "cap800": 800}
MIN_LISTED_DAYS = 120  # 上市不足此天数不入池（避开新股连续涨停阶段）


# ---------------------------------------------------------------- 转换


def _parse_date(col: pl.Expr) -> pl.Expr:
    """数据包里日期格式不统一（2015-09-14 / 20150914），统一解析为 Datetime。"""
    return col.cast(pl.Utf8).str.replace_all("-", "").str.strptime(pl.Datetime("us"), "%Y%m%d")


def _done(p: Path) -> bool:
    """下载器未完成时旁边有 .pfpart 标记文件。"""
    return p.exists() and not Path(str(p) + ".pfpart").exists()


def _read_csvs(zpath: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """读取 zip 内全部 CSV（跳过 macOS 垃圾文件和空文件），纵向拼接。"""
    parts = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or not name.endswith(".csv"):
                continue
            raw = z.read(name).removeprefix(b"\xef\xbb\xbf")  # 去掉 UTF-8 BOM
            if raw.strip():
                df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)  # 先全部按字符串读，统一后再转类型
                parts.append(df.select([c for c in columns if c in df.columns]) if columns else df)
    return pl.concat(parts, how="diagonal")


def convert(force: bool = False) -> None:
    DAILY_OUT.mkdir(parents=True, exist_ok=True)
    zips = {}
    for d in DAILY_DIRS:  # 前面的目录优先
        for p in sorted(d.glob("*.zip")) if d.exists() else []:
            zips.setdefault(p.stem, p)
    for year, p in sorted(zips.items()):
        out = DAILY_OUT / f"{year}.parquet"
        if out.exists() and not force:
            print(f"{year}: 已转换，跳过")
            continue
        if not _done(p):
            print(f"{year}: 未下载完成，跳过")
            continue
        t = time.time()
        try:
            df = _read_csvs(p, KEEP)
        except zipfile.BadZipFile:
            print(f"{year}: zip 损坏，跳过（重新下载后再运行）")
            continue
        df = df.with_columns(
            _parse_date(pl.col("datetime")),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in KEEP if c not in ("code", "datetime")],
        )
        df.write_parquet(out)
        print(f"{year}: {df['code'].n_unique()} 只, {len(df)} 行  ({time.time() - t:.0f}s)", flush=True)

    if (not ADJ_OUT.exists() or force) and _done(ADJ_ZIP):
        t = time.time()
        adj = _read_csvs(ADJ_ZIP)
        adj.columns = ["code", "datetime", "adj"]
        adj = adj.with_columns(_parse_date(pl.col("datetime")), pl.col("adj").cast(pl.Float64))
        adj.write_parquet(ADJ_OUT)
        print(f"复权因子: {adj['code'].n_unique()} 只  ({time.time() - t:.0f}s)")
    years = sorted(int(p.stem) for p in DAILY_OUT.glob("*.parquet"))
    missing = sorted(set(range(years[0], years[-1] + 1)) - set(years)) if years else []
    print(f"\n已转换年份: {years[0] if years else '-'}~{years[-1] if years else '-'}，缺失: {missing or '无'}")


def convert_names() -> None:
    """从分钟数据包提取 代码 -> 名称。按文件修改时间从新到旧，先出现的（最新的）为准。"""
    import unicodedata

    # 每个目录按文件名（年份/月份）从新到旧各取几个包，先出现的（最新的）名称为准
    zips = [p for d in MINUTE_DIRS if d.exists()
            for p in sorted((x for x in d.glob("*.zip") if _done(x)), key=lambda x: x.name, reverse=True)[:4]]
    names: dict[str, str] = {}
    for p in zips:
        try:
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.startswith("__MACOSX") or not n.endswith(".csv"):
                        continue
                    code = n.rsplit("/", 1)[-1].split("_")[0].split(".")[0]  # sh600519_2026.csv -> sh600519
                    if code in names:
                        continue
                    with z.open(n) as f:
                        f.readline()
                        row = f.readline().decode("utf-8-sig", "replace").split(",")
                    if len(row) > 2:
                        nm = unicodedata.normalize("NFKC", row[2]).replace(" ", "")  # 万  科Ａ -> 万科A
                        names[code] = nm
        except zipfile.BadZipFile:
            continue
    df = pl.DataFrame({"code": [f"{c[:2]}.{c[2:]}" for c in names], "name": list(names.values())})
    df.write_parquet(NAMES_OUT)
    print(f"名称映射: {len(df)} 只（来自 {[z.name for z in zips]}）")


# ---------------------------------------------------------------- 构建面板


def _daily_lazy(start: str, end: str) -> pl.LazyFrame:
    """惰性扫描已转换的日K：只读需要的年份和列，过滤下推到读取层。"""
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    files = [DAILY_OUT / f"{y}.parquet" for y in range(y0, y1 + 1)]
    missing = [f.stem for f in files if not f.exists()]
    if missing:
        print(f"⚠️ 缺少年份 {missing}：这些年份所有股票都视为停牌（回测会冻结持仓），请补齐后重新 convert + build")
    lf = pl.scan_parquet([f for f in files if f.exists()]).select(KEEP)
    return lf.filter(pl.col("datetime").is_between(pd.Timestamp(start), pd.Timestamp(end)))


def _dedupe(lf: pl.LazyFrame, what: str) -> pl.LazyFrame:
    """同一 (代码, 日期) 出现多次时保留最后一条，并报告数量与样例。"""
    dup = (lf.group_by("code", "datetime").len().filter(pl.col("len") > 1)
           .collect(engine="streaming"))
    if len(dup):
        print(f"⚠️ {what}: {len(dup)} 个 (代码, 日期) 重复，涉及 {dup['code'].n_unique()} 只，保留最后一条。"
              f"样例: {dup.head(3).rows()}")
    return lf.unique(["code", "datetime"], keep="last", maintain_order=True)


def _calendar(lf: pl.LazyFrame) -> pd.DatetimeIndex:
    """交易日历 = 数据包日期 ∪ 基准指数日期（缺失年份会表现为"全部停牌"而不是时间轴断开）。"""
    dates = set(lf.select(pl.col("datetime").unique()).collect(engine="streaming")["datetime"].to_list())
    bench = config.RAW_DIR / f"{config.BENCHMARK}.pkl"
    if bench.exists():
        b = pd.read_pickle(bench)["date"]
        dates |= set(b[(b >= config.START_DATE) & (b <= config.END_DATE)])
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates))


def cap_membership(lf: pl.LazyFrame, calendar: pd.DatetimeIndex, n: int) -> pd.DataFrame:
    """在长表上计算：每月末按流通市值取前 n（需上市满 MIN_LISTED_DAYS 个自然日×1.5），向前填充。

    返回宽表 bool（index=日历, columns=曾经入池的股票）。不构建全市场宽表，内存占用小。
    """
    from quant.backtest import month_ends

    snaps = month_ends(calendar)
    # 上市日：数据包里首次出现的日期（扫描全部年份，只读两列）
    first = (pl.scan_parquet(sorted(DAILY_OUT.glob("*.parquet"))).group_by("code")
             .agg(pl.col("datetime").min().alias("first_day")))
    min_age = pl.duration(days=int(MIN_LISTED_DAYS * 1.5))  # 交易日 -> 自然日近似
    top = (lf.filter(pl.col("datetime").is_in(list(snaps)) & (pl.col("circ_mv") > 0))
           .join(first, on="code")
           .filter(pl.col("datetime") - pl.col("first_day") >= min_age)
           .with_columns(pl.col("circ_mv").rank("ordinal", descending=True).over("datetime").alias("rk"))
           .filter(pl.col("rk") <= n)
           .select("datetime", "code")
           .collect(engine="streaming").to_pandas())
    snap = pd.crosstab(top["datetime"], top["code"]).astype(bool)
    snap = snap.reindex(snaps).fillna(False)  # 某月末全部缺数据（缺失年份）时该月为空池
    return snap.reindex(calendar).ffill().fillna(False).astype(bool)


def build(universe: str = config.UNIVERSE) -> dict:
    from quant.data import PANEL_FILE

    if universe not in CAP_UNIVERSES:
        raise ValueError(f"本地数据源只支持 {list(CAP_UNIVERSES)}，当前 UNIVERSE={universe}")
    t = time.time()
    lf = _dedupe(_daily_lazy(config.START_DATE, config.END_DATE), "日K")
    calendar = _calendar(lf)

    # 1) 股票池：长表上排名，只得到"曾经入池"的股票列表
    member = cap_membership(lf, calendar, CAP_UNIVERSES[universe])
    codes = list(member.columns)
    print(f"股票池 {universe}: 曾入池 {len(codes)} 只  ({time.time() - t:.0f}s)")

    # 2) 只对这些股票合并复权因子、换算单位，然后才物化到内存
    adj = _dedupe(pl.scan_parquet(ADJ_OUT), "复权因子")
    long = (lf.filter(pl.col("code").is_in(codes))
            .join(adj.filter(pl.col("code").is_in(codes)), on=["code", "datetime"], how="left")
            .sort("code", "datetime")
            .with_columns(pl.col("adj").forward_fill().over("code").fill_null(1.0))  # 缺因子沿用前值
            .with_columns(
                *[(pl.col(c) * pl.col("adj")).alias(c) for c in ["open", "high", "low", "close"]],
                pl.col("volume") * 100,  # 手 -> 股
                pl.col("amount") * 1000,  # 千元 -> 元
                pl.col("circ_mv") * 1e4,  # 万元 -> 元
            )
            .collect(engine="streaming").to_pandas())
    print(f"已读入 {len(long)} 行  ({time.time() - t:.0f}s)")
    # 代码统一成 baostock 格式（600519.SH -> sh.600519），与项目其它部分一致
    to_bs = lambda c: f"{c[-2:].lower()}.{c[:-3]}"  # noqa: E731
    long["code"] = long["code"].map(to_bs)
    member = member.rename(columns=to_bs)
    codes = list(member.columns)

    # 3) 转成与 baostock 版本同结构的宽表面板（此时数据量已很小）
    long = long.set_index(["datetime", "code"])

    def wide(field: str) -> pd.DataFrame:
        return long[field].unstack("code").reindex(index=calendar, columns=codes)

    panel = {f: wide(f) for f in ["open", "high", "low", "close", "volume", "amount", "circ_mv"]}
    panel["turn"] = wide("turnover")
    panel["peTTM"] = wide("pe_ttm")
    panel["pbMRQ"] = wide("pb")
    panel["dv_ttm"] = wide("dv_ttm")
    del long
    has_bar = panel["open"].notna() & (panel["volume"] > 0)
    panel["can_trade"] = has_bar  # 数据包里当天没有行情 = 停牌
    panel["member"] = member
    panel["tradable"] = has_bar & member  # 无 ST 标记：大市值池中 ST 很少，暂不处理
    panel["close"] = panel["close"].ffill()

    # 指数：数据包没有指数，用 baostock 已缓存的指数日线（data/raw/）
    idx = [pd.read_pickle(p) for c in config.INDICES if (p := config.RAW_DIR / f"{c}.pkl").exists()]
    if idx:
        ind = pd.concat(idx)
        panel["index_close"] = ind.pivot(index="date", columns="code", values="close").reindex(calendar).ffill()
        panel["index_open"] = ind.pivot(index="date", columns="code", values="open").reindex(calendar)
    else:
        print("⚠️ 未找到指数缓存 data/raw/sh.000300.pkl 等，基准将不可用")
    if NAMES_OUT.exists():  # 当前名称（仅用于显示）
        panel["names"] = pd.read_parquet(NAMES_OUT).set_index("code")["name"]
    else:
        panel["names"] = pd.Series(dtype=str)
        print("⚠️ 无名称映射，可运行: uv run python -m quant.local_data names")

    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, PANEL_FILE)
    n_member = member.sum(axis=1)
    print(f"面板已保存: {PANEL_FILE}")
    print(f"  {len(calendar)} 个交易日 × {len(codes)} 只；每日成分数 {n_member.min()}~{n_member.max()}"
          f"  ({time.time() - t:.0f}s)")
    return panel


def main() -> None:
    p = argparse.ArgumentParser(description="购买的本地数据包 -> 项目面板")
    p.add_argument("step", choices=["convert", "names", "build", "all"])
    p.add_argument("--force", action="store_true", help="convert 时重新转换已完成的年份")
    a = p.parse_args()
    if a.step in ("convert", "all"):
        convert(a.force)
    if a.step in ("names", "all") and (a.force or not NAMES_OUT.exists() or a.step == "names"):
        convert_names()
    if a.step in ("build", "all"):
        build()


if __name__ == "__main__":
    main()
