"""执行仿真：用事件驱动框架 backtrader 重放策略的目标权重，显式建模 A 股执行摩擦。

向量化回测（vectorbt）假设"信号日次日开盘全部成交 + 固定 0.2% 成本"。这里换成 backtrader 的
事件循环（bar 逐根推进，订单 submit -> accept -> (partial) -> completed/expired），只通过它的
标准扩展点注入 A 股规则，不自己写撮合循环：

| 扩展点                         | 建模内容                                                         |
|-------------------------------|------------------------------------------------------------------|
| PandasData 附加 lines          | 成交额、20 日 ADV、20 日波动率（均滞后一天）、能否成交、每手对应的复权单位 |
| broker filler（AShareFiller）  | 停牌不成交；开盘涨停买不进 / 跌停卖不出；单日成交 ≤ 当日成交额 × 参与率上限；整手 |
| CommInfoBase（AShareCommission）| 佣金(最低 5 元) + 过户费 + 印花税(2023-08-28 起减半) + 半个价差 + 平方根冲击 |
| Strategy（TargetWeights）       | 每个信号日收盘按目标权重下单（先卖后买、整手、有效期 valid_days）          |

说明：
- 价差与冲击作为"额外成本"计入 commission（按成交额扣现金），经济上等价于成交价变差，
  且便于分项统计；成交价本身用开盘价（无摩擦参考价）。
- 未成交部分（涨跌停、停牌、超过参与率上限）在有效期内逐日继续撮合，过期作废。
- 平方根冲击：impact = eta * sigma_daily * sqrt(成交额 / ADV)。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field

import backtrader as bt
import numpy as np
import pandas as pd

from quant import backtest as vb
from quant import config

STAMP_DUTY_CUT = dt.date(2023, 8, 28)
ETF_ADV_RATIO = 0.01  # 指数类资产用 ETF 交易：ETF 日成交额 ≈ 指数成交额 × 1%（沪深300ETF 量级）
ETF_PRICE_DIV = 1000  # ETF 价格 ≈ 指数点位 / 1000


# ---------------------------------------------------------------- 参数


@dataclass
class ExecConfig:
    cash: float = 10_000_000
    commission: float = 0.00025
    min_commission: float = 5.0
    transfer_fee: float = 0.00001
    stamp_old: float = 0.001
    stamp_new: float = 0.0005
    tick: float = 0.01  # 股票最小报价单位（元）
    etf_tick: float = 0.001  # ETF 最小报价单位（元）
    min_half_spread: float = 0.0002
    impact_eta: float = 0.7
    participation: float = 0.10  # 单日最多成交当日成交额的 10%
    lot: int = 100
    valid_days: int = 5  # 未成交订单有效期（交易日）
    cash_buffer: float = 0.01  # 预留现金，覆盖成本，避免买单因资金不足被拒
    min_trade_frac: float = 0.002  # 小于总资产 0.2% 的调整不做
    limits: bool = True  # 涨跌停约束
    costs: bool = True  # False = 零成本（用于和 vectorbt 对账）

    @classmethod
    def frictionless(cls, cash: float = 10_000_000) -> "ExecConfig":
        return cls(cash=cash, commission=0, min_commission=0, transfer_fee=0, stamp_old=0, stamp_new=0,
                   tick=0, etf_tick=0, min_half_spread=0, impact_eta=0, participation=1e9, lot=1, limits=False, costs=False,
                   min_trade_frac=0, cash_buffer=0.01)


@dataclass
class Ledger:
    """成本流水（由策略 notify_order 在每笔成交时写入）与撮合受阻计数（由 filler 写入）。"""
    costs: list = field(default_factory=list)
    blocked: dict = field(default_factory=lambda: defaultdict(int))


# ---------------------------------------------------------------- 数据


class AShareData(bt.feeds.PandasData):
    lines = ("amount", "adv", "sigma", "can_trade", "lot_units", "raw_px", "limit_pct")
    params = tuple((ln, -1) for ln in lines)


def _limit_pct(code: str, index: pd.DatetimeIndex) -> pd.Series:
    """涨跌幅限制：科创板 20%；创业板 2020-08-24 起 20%；其余 10%（ST 已从股票池剔除）。"""
    if code.startswith("sh.688"):
        return pd.Series(0.2, index=index)
    if code.startswith("sz.30"):
        return pd.Series(np.where(index >= "2020-08-24", 0.2, 0.1), index=index)
    return pd.Series(0.1, index=index)


def stock_frame(panel: dict, code: str, lot: int) -> pd.DataFrame:
    close, open_ = vb._exec_prices(panel)
    idx = close.index
    can = vb.exec_mask(panel)[code].reindex(idx).fillna(False)
    o = open_[code]
    h = panel["high"][code].reindex(idx).fillna(o)
    l_ = panel["low"][code].reindex(idx).fillna(o)
    amount = panel["amount"][code].reindex(idx).fillna(0.0).where(can, 0.0)
    volume = panel["volume"][code].reindex(idx)
    # 实际（不复权）成交均价 = 成交额 / 成交量；复权价 / 实际价 = 复权因子，用来把"一手=100股"换成复权单位
    raw = (amount / volume).where(volume > 0)
    typical = (o + h + l_ + close[code]) / 4
    raw = raw.ffill().bfill()
    factor = (typical / raw).rolling(5, min_periods=1).median()
    ret = close[code].pct_change(fill_method=None)
    df = pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": close[code], "volume": panel["volume"][code].fillna(0),
        "amount": amount,
        "adv": amount.where(can).rolling(20, min_periods=5).mean().shift(1).fillna(0),
        "sigma": ret.rolling(20, min_periods=5).std().shift(1).fillna(0.02),
        "can_trade": can.astype(float),
        "lot_units": (lot / factor.shift(1)).bfill(),  # 一手 = 多少复权单位
        "raw_px": raw.shift(1).bfill(),
        "limit_pct": _limit_pct(code, idx),
    }, index=idx)
    return df


def etf_frame(code: str, index: pd.DatetimeIndex, lot: int) -> pd.DataFrame:
    """指数 -> ETF 近似：价格按指数点位，成交额按比例缩小，免印花税，不设涨跌停（10% 内很少触发）。"""
    raw = pd.read_pickle(config.RAW_DIR / f"{code}.pkl").set_index("date").reindex(index)
    raw = raw.ffill()
    amount = raw["amount"] * ETF_ADV_RATIO
    ret = raw["close"].pct_change()
    return pd.DataFrame({
        "open": raw["open"].fillna(raw["close"]), "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "volume": raw["volume"] * ETF_ADV_RATIO,
        "amount": amount,
        "adv": amount.rolling(20, min_periods=5).mean().shift(1).bfill(),
        "sigma": ret.rolling(20, min_periods=5).std().shift(1).fillna(0.015),
        "can_trade": 1.0,
        "lot_units": float(lot) / ETF_PRICE_DIV,
        "raw_px": raw["close"].shift(1).bfill() / ETF_PRICE_DIV,
        "limit_pct": 1.0,
    }, index=index)


# ---------------------------------------------------------------- 成本


def cost_parts(cfg: ExecConfig, d, size: float, price: float, is_etf: bool) -> dict:
    """一笔成交的各项成本（元）。size>0 买，<0 卖；d 为当前 bar 位于成交日的 data。"""
    value = abs(size) * price
    if not cfg.costs or value == 0:
        return {"佣金": 0.0, "过户费": 0.0, "印花税": 0.0, "价差": 0.0, "冲击": 0.0}
    date = d.datetime.date(0)
    tax = 0.0
    if size < 0 and not is_etf:
        tax = value * (cfg.stamp_old if date < STAMP_DUTY_CUT else cfg.stamp_new)
    raw_px = d.raw_px[0]
    tick = cfg.etf_tick if is_etf else cfg.tick
    half_spread = max(tick / 2 / raw_px, cfg.min_half_spread) if raw_px > 0 else cfg.min_half_spread
    adv, sigma = d.adv[0], d.sigma[0]
    impact = cfg.impact_eta * sigma * math.sqrt(value / adv) if adv > 0 else 0.0
    return {
        "佣金": max(value * cfg.commission, cfg.min_commission),
        "过户费": 0.0 if is_etf else value * cfg.transfer_fee,
        "印花税": tax,
        "价差": value * half_spread,
        "冲击": value * impact,
    }


class AShareCommission(bt.CommInfoBase):
    """把显性费用和隐性成本（价差+冲击）统一作为"佣金"从现金中扣除。"""

    params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_FIXED), ("cfg", None), ("is_etf", False))

    def _getcommission(self, size, price, pseudoexec):
        return sum(cost_parts(self.p.cfg, self.data, size, price, self.p.is_etf).values())


class AShareFiller:
    """决定本根 bar 能成交多少（backtrader 的 volume filler 接口）。"""

    def __init__(self, cfg: ExecConfig, ledger: Ledger):
        self.cfg, self.ledger = cfg, ledger

    def __call__(self, order, price, ago):
        d, cfg = order.data, self.cfg
        rem = abs(order.executed.remsize)
        if not d.can_trade[ago]:
            self.ledger.blocked["停牌"] += 1
            return 0
        if cfg.limits and len(d) > 1:
            prev, lim = d.close[ago - 1], d.limit_pct[ago]
            if order.isbuy() and d.open[ago] >= prev * (1 + lim) * 0.998:
                self.ledger.blocked["涨停买不进"] += 1
                return 0
            if order.issell() and d.open[ago] <= prev * (1 - lim) * 1.002:
                self.ledger.blocked["跌停卖不出"] += 1
                return 0
        cap = cfg.participation * d.amount[ago] / price if price > 0 else 0
        size = min(rem, cap)
        if size < rem:
            self.ledger.blocked["超参与率(部分成交)"] += 1
        lot = d.lot_units[ago] if cfg.lot > 1 else 0
        if lot > 0 and size < rem:  # 部分成交按整手；全部成交（含清仓零股）不取整
            size = math.floor(size / lot) * lot
        return size


# ---------------------------------------------------------------- 策略


class TargetWeights(bt.Strategy):
    """在信号日收盘按目标权重下单，次日开盘起撮合。"""

    params = (("weights", None), ("mode", "full"), ("cfg", None), ("ledger", None), ("is_etf", False))

    def __init__(self):
        w: pd.DataFrame = self.p.weights
        self.signals = {d.date(): row for d, row in w.iterrows()}
        self.by_name = {d._name: d for d in self.datas}
        self.values, self.expired = [], 0
        self.open_orders: dict[str, list] = defaultdict(list)

    def notify_order(self, order):
        if order.status in (order.Completed, order.Partial):
            d = order.data
            for eb in order.executed.iterpending():  # 本次通知新增的成交片段
                parts = cost_parts(self.p.cfg, d, eb.size, eb.price, self.p.is_etf)
                value = abs(eb.size) * eb.price
                self.p.ledger.costs.append({
                    "date": pd.Timestamp(d.datetime.date(0)), "code": d._name, "side": "买" if eb.size > 0 else "卖",
                    "value": value, **parts,
                    "参与率": value / d.amount[0] if d.amount[0] > 0 else np.nan,
                    "开盘偏离": eb.price / d.close[-1] - 1,  # 相对信号日收盘的隔夜跳空
                })
        if order.status in (order.Expired, order.Margin, order.Rejected) and order.executed.remsize:
            self.expired += 1
        if not order.alive():
            lst = self.open_orders[order.data._name]
            if order in lst:
                lst.remove(order)

    def next(self):
        today = self.datas[0].datetime.date(0)
        self.values.append((pd.Timestamp(today), self.broker.getvalue()))
        row = self.signals.get(today)
        if row is None:
            return
        cfg: ExecConfig = self.p.cfg
        equity = self.broker.getvalue()
        investable = equity * (1 - cfg.cash_buffer)
        valid = today + dt.timedelta(days=int(cfg.valid_days * 1.5) + 1)
        sells, buys = [], []
        names = row.index if self.p.mode == "hold" else self.by_name.keys()
        for name in names:
            d = self.by_name.get(name)
            if d is None:
                continue
            w = row.get(name, np.nan)
            if self.p.mode == "hold" and pd.isna(w):
                continue
            w = 0.0 if pd.isna(w) else float(w)
            for o in list(self.open_orders[name]):  # 新信号覆盖旧的未完成订单
                self.cancel(o)
            pos = self.getposition(d).size
            px = d.close[0]
            delta = w * investable / px - pos
            if w == 0 and pos > 0:
                sells.append((d, -pos))  # 清仓（允许零股）
                continue
            if abs(delta) * px < cfg.min_trade_frac * equity:
                continue
            lot = d.lot_units[0] if cfg.lot > 1 else 0
            if lot > 0:
                delta = math.copysign(math.floor(abs(delta) / lot) * lot, delta)
            if delta == 0:
                continue
            (buys if delta > 0 else sells).append((d, delta))
        for d, s in sells + buys:  # 先卖后买：同一根 bar 内按提交顺序撮合
            o = (self.buy if s > 0 else self.sell)(data=d, size=abs(s), exectype=bt.Order.Market, valid=valid)
            self.open_orders[d._name].append(o)


# ---------------------------------------------------------------- 运行


@dataclass
class ExecResult:
    value: pd.Series
    costs: pd.DataFrame
    blocked: pd.Series
    expired: int
    cfg: ExecConfig

    def stats(self) -> pd.Series:
        s = vb.perf_stats(self.value)
        years = len(self.value) / vb.TRADING_DAYS
        avg_eq = self.value.mean()
        c = self.costs
        traded = c["value"].sum() if len(c) else 0.0
        for k in ["佣金", "过户费", "印花税", "价差", "冲击"]:
            s[f"{k}(年化bp)"] = (c[k].sum() if len(c) else 0.0) / avg_eq / years * 1e4
        s["总成本(年化bp)"] = sum(s[f"{k}(年化bp)"] for k in ["佣金", "过户费", "印花税", "价差", "冲击"])
        s["年化双边换手"] = traded / avg_eq / years
        s["平均参与率"] = c["参与率"].mean() if len(c) else np.nan
        s["受阻撮合次数"] = self.blocked.sum()
        s["过期/拒绝订单"] = self.expired
        return s


def run(panel: dict, weights: pd.DataFrame, mode: str = "full", asset: str = "stock",
        cfg: ExecConfig | None = None, start: str = config.BACKTEST_START) -> ExecResult:
    """用 backtrader 执行一组目标权重。weights 的 index 是信号日（收盘后决定）。"""
    cfg = cfg or ExecConfig()
    ledger = Ledger()
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=False)
    cerebro.broker.setcash(cfg.cash)
    cerebro.broker.set_checksubmit(False)  # 资金在撮合时检查（卖单先成交释放资金）
    cerebro.broker.set_filler(AShareFiller(cfg, ledger))

    # 起点前一个交易日开始喂数据，使起点前最后一个信号在起点当天开盘执行（与 vectorbt 版一致）
    idx = panel["close"].index
    first = idx.searchsorted(pd.Timestamp(start))
    idx = idx[max(first - 1, 0):]
    w = weights.loc[weights.index >= idx[0]]
    prior = weights.loc[weights.index < idx[0]]
    if len(prior) and (len(w) == 0 or w.index[0] != idx[0]):
        w = pd.concat([prior.iloc[[-1]].set_axis([idx[0]]), w])
    codes = [c for c in w.columns if (w[c].fillna(0) != 0).any()]

    for code in codes:
        if asset == "index":
            df = etf_frame(code, idx, cfg.lot)
        else:
            df = stock_frame(panel, code, cfg.lot).loc[idx]
        data = AShareData(dataname=df, name=code)
        cerebro.adddata(data)
        cerebro.broker.addcommissioninfo(
            AShareCommission(cfg=cfg, is_etf=(asset == "index")), name=code)
    # CommInfo 需要拿到对应的 data（用于读取当日成交额、波动、日期）
    for data in cerebro.datas:
        cerebro.broker.getcommissioninfo(data).data = data

    cerebro.addstrategy(TargetWeights, weights=w[codes], mode=mode, cfg=cfg, ledger=ledger, is_etf=(asset == "index"))
    strat = cerebro.run(runonce=True, preload=True)[0]

    value = pd.Series(dict(strat.values)).loc[start:]
    costs = pd.DataFrame(ledger.costs)
    if len(costs):
        costs = costs.loc[costs["date"] >= pd.Timestamp(start)]
    return ExecResult(value=value, costs=costs, blocked=pd.Series(dict(ledger.blocked), dtype=float),
                      expired=strat.expired, cfg=cfg)
