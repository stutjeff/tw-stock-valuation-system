import os
import json
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "台股錯殺・轉機・護城河估值系統"
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")


# -----------------------------
# Google Trends helpers
# -----------------------------

GOOGLE_TRENDS_REGION_MAP = {
    "台灣": {"geo": "TW", "pn": "taiwan", "hl": "zh-TW", "tz": 480},
    "美國": {"geo": "US", "pn": "united_states", "hl": "en-US", "tz": -300},
}


def clean_keywords(raw: str, limit: int = 5) -> List[str]:
    parts: List[str] = []
    for chunk in raw.replace("，", ",").replace("\n", ",").split(","):
        kw = chunk.strip()
        if kw and kw not in parts:
            parts.append(kw)
    return parts[:limit]


@st.cache_data(ttl=1800, show_spinner=False)
def trends_hot_searches(region_label: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if TrendReq is None:
        return pd.DataFrame(), "尚未安裝 pytrends，請確認 requirements.txt 內有 pytrends。"
    cfg = GOOGLE_TRENDS_REGION_MAP.get(region_label, GOOGLE_TRENDS_REGION_MAP["台灣"])
    try:
        pytrends = TrendReq(hl=cfg["hl"], tz=cfg["tz"], timeout=(10, 25), retries=1, backoff_factor=0.2)
        df = pytrends.trending_searches(pn=cfg["pn"])
        if df is None or df.empty:
            return pd.DataFrame(), "Google Trends 目前沒有回傳熱門搜尋資料。"
        out = df.reset_index(drop=True)
        out.columns = ["熱門搜尋"]
        out.insert(0, "排名", range(1, len(out) + 1))
        return out, None
    except Exception as e:
        return pd.DataFrame(), f"Google Trends 熱門搜尋讀取失敗：{e}"


@st.cache_data(ttl=1800, show_spinner=False)
def trends_interest_over_time(keywords: Tuple[str, ...], geo: str, timeframe: str, hl: str, tz: int) -> Tuple[pd.DataFrame, Optional[str]]:
    if TrendReq is None:
        return pd.DataFrame(), "尚未安裝 pytrends，請確認 requirements.txt 內有 pytrends。"
    try:
        kw_list = list(keywords)[:5]
        if not kw_list:
            return pd.DataFrame(), "請輸入至少一個關鍵字。"
        pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 25), retries=1, backoff_factor=0.2)
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop="")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return pd.DataFrame(), "Google Trends 沒有回傳時間序列資料，可能是搜尋量太低或暫時被限制。"
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.reset_index().rename(columns={"date": "日期"})
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Google Trends 關鍵字趨勢讀取失敗：{e}"


@st.cache_data(ttl=1800, show_spinner=False)
def trends_related_queries(keyword: str, geo: str, timeframe: str, hl: str, tz: int) -> Tuple[Dict[str, Any], Optional[str]]:
    if TrendReq is None:
        return {}, "尚未安裝 pytrends，請確認 requirements.txt 內有 pytrends。"
    try:
        pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 25), retries=1, backoff_factor=0.2)
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
        data = pytrends.related_queries()
        return data or {}, None
    except Exception as e:
        return {}, f"Google Trends 相關搜尋讀取失敗：{e}"


def plot_trends_interest(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df.empty or "日期" not in df.columns:
        return fig
    for col in df.columns:
        if col != "日期":
            fig.add_trace(go.Scatter(x=df["日期"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(height=380, title=title, yaxis_title="搜尋熱度（0-100，相對值）", margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -----------------------------
# Helpers
# -----------------------------

def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return os.environ.get(name, default)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", "").replace("--", "").strip()
            if x in ["", "-", "nan", "NaN", "None"]:
                return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in [None, 0]:
        return None
    try:
        return (a / b - 1) * 100
    except Exception:
        return None


def fmt_num(x: Any, digits: int = 2, suffix: str = "") -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{v:,.{digits}f}{suffix}"


def fmt_int(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{int(round(v)):,.0f}"


def normalize_stock_id(s: str) -> str:
    return "".join(ch for ch in s.strip() if ch.isalnum()).upper()


@st.cache_data(ttl=3600, show_spinner=False)
def finmind(dataset: str, data_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    token = get_secret("FINMIND_TOKEN")
    if token:
        params["token"] = token
    try:
        r = requests.get(FINMIND_URL, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"FinMind 讀取 {dataset} 失敗：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def stock_info() -> pd.DataFrame:
    df = finmind("TaiwanStockInfo")
    if df.empty:
        return df
    # Common columns: stock_id, stock_name, industry_category, type, date
    return df


def get_company_row(stock_id: str) -> Dict[str, Any]:
    df = stock_info()
    if df.empty:
        return {"stock_id": stock_id, "stock_name": "", "industry_category": "", "type": ""}
    row = df[df["stock_id"].astype(str) == str(stock_id)]
    if row.empty:
        return {"stock_id": stock_id, "stock_name": "", "industry_category": "", "type": ""}
    return row.iloc[0].to_dict()


def years_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def add_bollinger(df: pd.DataFrame, windows=(20, 60, 240)) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    for w in windows:
        ma = out["close"].rolling(w).mean()
        sd = out["close"].rolling(w).std()
        out[f"MA{w}"] = ma
        out[f"U1_{w}"] = ma + sd
        out[f"L1_{w}"] = ma - sd
        out[f"U2_{w}"] = ma + 2 * sd
        out[f"L2_{w}"] = ma - 2 * sd
        out[f"Z_{w}"] = (out["close"] - ma) / sd
    return out


def latest_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def get_price(stock_id: str, days: int = 900) -> pd.DataFrame:
    df = finmind("TaiwanStockPrice", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    for col in ["open", "max", "min", "close", "Trading_Volume", "Trading_money"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return add_bollinger(df)


def get_per_pbr(stock_id: str, days: int = 900) -> pd.DataFrame:
    df = finmind("TaiwanStockPER", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col not in ["date", "stock_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date")


def get_month_revenue(stock_id: str, months_back: int = 48) -> pd.DataFrame:
    df = finmind("TaiwanStockMonthRevenue", stock_id, years_ago(months_back * 31), date.today().isoformat())
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col not in ["date", "stock_id", "country"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date")
    rev_col = "revenue" if "revenue" in df.columns else None
    if rev_col:
        df["revenue_yoy"] = df[rev_col].pct_change(12) * 100
        df["revenue_mom"] = df[rev_col].pct_change(1) * 100
        df["revenue_3m_yoy_avg"] = df["revenue_yoy"].rolling(3).mean()
    return df


def get_institutional(stock_id: str, days: int = 90) -> pd.DataFrame:
    df = finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col not in ["date", "stock_id", "name"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date")


def pivot_inst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "name" not in df.columns:
        return pd.DataFrame()
    net_col = None
    for c in ["buy", "sell", "buy_sell", "Buy", "Sell"]:
        pass
    if "buy" in df.columns and "sell" in df.columns:
        tmp = df.copy()
        tmp["net"] = tmp["buy"] - tmp["sell"]
    elif "buy_sell" in df.columns:
        tmp = df.copy()
        tmp["net"] = tmp["buy_sell"]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        tmp = df.copy()
        tmp["net"] = tmp[numeric_cols[-1]]
    p = tmp.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
    p["合計"] = p.sum(axis=1)
    return p.reset_index()


def get_margin(stock_id: str, days: int = 180) -> pd.DataFrame:
    df = finmind("TaiwanStockMarginPurchaseShortSale", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col not in ["date", "stock_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date")


def get_financials(stock_id: str, days: int = 1700) -> Dict[str, pd.DataFrame]:
    start = years_ago(days)
    end = date.today().isoformat()
    return {
        "income": finmind("TaiwanStockFinancialStatements", stock_id, start, end),
        "balance": finmind("TaiwanStockBalanceSheet", stock_id, start, end),
        "cashflow": finmind("TaiwanStockCashFlowsStatement", stock_id, start, end),
        "dividend": finmind("TaiwanStockDividend", stock_id, start, end),
    }


def tidy_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    for col in out.columns:
        if col not in ["date", "stock_id", "type", "origin_name"]:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out.sort_values("date") if "date" in out.columns else out


def extract_metric(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    if df.empty:
        return None
    name_cols = [c for c in ["type", "origin_name"] if c in df.columns]
    val_cols = [c for c in ["value"] if c in df.columns]
    if not name_cols or not val_cols:
        return None
    latest_date = df["date"].max() if "date" in df.columns else None
    sub = df[df["date"] == latest_date] if latest_date is not None else df
    for cand in candidates:
        mask = pd.Series(False, index=sub.index)
        for nc in name_cols:
            mask = mask | sub[nc].astype(str).str.contains(cand, case=False, na=False)
        m = sub[mask]
        if not m.empty:
            return safe_float(m.iloc[-1]["value"])
    return None


def financial_summary(fin: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    income = tidy_statement(fin.get("income", pd.DataFrame()))
    balance = tidy_statement(fin.get("balance", pd.DataFrame()))
    cashflow = tidy_statement(fin.get("cashflow", pd.DataFrame()))

    net_income = extract_metric(income, ["本期淨利", "稅後淨利", "ProfitLoss", "NetIncome"])
    revenue = extract_metric(income, ["營業收入", "Revenue"])
    gross_profit = extract_metric(income, ["營業毛利", "GrossProfit"])
    operating_income = extract_metric(income, ["營業利益", "OperatingIncome"])
    eps = extract_metric(income, ["基本每股盈餘", "EPS", "EarningsPerShare"])

    total_equity = extract_metric(balance, ["權益總計", "Equity", "TotalEquity"])
    total_assets = extract_metric(balance, ["資產總計", "TotalAssets"])
    total_liabilities = extract_metric(balance, ["負債總計", "Liabilities", "TotalLiabilities"])

    ocf = extract_metric(cashflow, ["營業活動之淨現金流入", "CashFlowsFromUsedInOperatingActivities", "OperatingActivities"])
    capex = extract_metric(cashflow, ["取得不動產", "PropertyPlantAndEquipment", "CapitalExpenditure"])

    roe = (net_income / total_equity * 100) if net_income is not None and total_equity not in [None, 0] else None
    roa = (net_income / total_assets * 100) if net_income is not None and total_assets not in [None, 0] else None
    debt_ratio = (total_liabilities / total_assets * 100) if total_liabilities is not None and total_assets not in [None, 0] else None
    gross_margin = (gross_profit / revenue * 100) if gross_profit is not None and revenue not in [None, 0] else None
    op_margin = (operating_income / revenue * 100) if operating_income is not None and revenue not in [None, 0] else None
    fcf = None
    if ocf is not None:
        # capex may be negative depending on data; keep conservative display only
        fcf = ocf + capex if capex is not None and capex < 0 else ocf - abs(capex or 0)

    return {
        "revenue": revenue,
        "net_income": net_income,
        "eps": eps,
        "roe": roe,
        "roa": roa,
        "debt_ratio": debt_ratio,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "ocf": ocf,
        "fcf": fcf,
    }


def percentile_rank(series: pd.Series, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 20:
        return None
    return (s <= value).mean() * 100


def score_and_classify(company: Dict[str, Any], price: pd.DataFrame, per: pd.DataFrame, rev: pd.DataFrame, inst_p: pd.DataFrame, fs: Dict[str, Any]) -> Dict[str, Any]:
    latest_price = latest_row(price)
    latest_per = latest_row(per)

    close = safe_float(latest_price.get("close"))
    z240 = safe_float(latest_price.get("Z_240"))
    pe = safe_float(latest_per.get("PER")) or safe_float(latest_per.get("本益比"))
    pbr = safe_float(latest_per.get("PBR")) or safe_float(latest_per.get("股價淨值比"))
    div_yield = safe_float(latest_per.get("dividend_yield")) or safe_float(latest_per.get("殖利率"))

    pe_rank = percentile_rank(per["PER"] if "PER" in per.columns else pd.Series(dtype=float), pe)
    pbr_rank = percentile_rank(per["PBR"] if "PBR" in per.columns else pd.Series(dtype=float), pbr)

    latest_rev = latest_row(rev)
    rev_yoy = safe_float(latest_rev.get("revenue_yoy"))
    rev_3m = safe_float(latest_rev.get("revenue_3m_yoy_avg"))

    inst_5 = inst_20 = None
    if not inst_p.empty and "合計" in inst_p.columns:
        inst_5 = inst_p.tail(5)["合計"].sum()
        inst_20 = inst_p.tail(20)["合計"].sum()

    # Scores
    industry_score = 8  # rule-based default; user adjusts with qualitative AI
    moat_score = 0
    industry = str(company.get("industry_category", ""))
    moat_keywords = ["半導體", "電機", "電子", "通信", "水泥", "食品", "金融", "生技", "環保", "綠能", "其他電子"]
    if any(k in industry for k in moat_keywords):
        moat_score += 7
    else:
        moat_score += 5
    if fs.get("roe") is not None and fs["roe"] >= 12:
        moat_score += 4
    if fs.get("gross_margin") is not None and fs["gross_margin"] >= 25:
        moat_score += 3
    moat_score = min(15, moat_score)

    financial_score = 0
    if fs.get("roe") is not None:
        financial_score += 7 if fs["roe"] >= 12 else 5 if fs["roe"] >= 8 else 2 if fs["roe"] > 0 else 0
    if fs.get("debt_ratio") is not None:
        financial_score += 5 if fs["debt_ratio"] < 45 else 3 if fs["debt_ratio"] < 65 else 1
    if fs.get("op_margin") is not None:
        financial_score += 4 if fs["op_margin"] >= 10 else 2 if fs["op_margin"] > 0 else 0
    if fs.get("fcf") is not None:
        financial_score += 4 if fs["fcf"] > 0 else 1
    financial_score = min(20, financial_score)

    growth_score = 0
    if rev_yoy is not None:
        growth_score += 8 if rev_yoy >= 20 else 6 if rev_yoy >= 10 else 4 if rev_yoy >= 0 else 1
    if rev_3m is not None:
        growth_score += 7 if rev_3m >= 15 else 5 if rev_3m >= 5 else 3 if rev_3m >= 0 else 0
    growth_score = min(15, growth_score)

    valuation_score = 0
    if pbr_rank is not None:
        valuation_score += 7 if pbr_rank <= 25 else 5 if pbr_rank <= 50 else 2 if pbr_rank <= 75 else 0
    elif pbr is not None:
        valuation_score += 5 if pbr <= 1.5 else 3 if pbr <= 2.5 else 1
    if pe_rank is not None:
        valuation_score += 5 if pe_rank <= 30 else 3 if pe_rank <= 60 else 1
    elif pe is not None and pe > 0:
        valuation_score += 5 if pe <= 15 else 3 if pe <= 25 else 1
    if z240 is not None:
        valuation_score += 3 if z240 <= -1 else 2 if z240 <= 0 else 0
    valuation_score = min(15, valuation_score)

    chip_score = 0
    if inst_20 is not None:
        chip_score += 5 if inst_20 > 0 else 2
    if inst_5 is not None:
        chip_score += 5 if inst_5 > 0 else 2
    chip_score = min(10, chip_score)

    tech_score = 0
    if z240 is not None:
        tech_score = 5 if -2 <= z240 <= 0.5 else 3 if z240 < -2 else 1 if z240 > 2 else 2

    governance_score = 3  # placeholder unless user adds internal-holder data later
    total = industry_score + moat_score + financial_score + growth_score + valuation_score + chip_score + tech_score + governance_score

    wrong_kill = (
        financial_score >= 12 and valuation_score >= 8 and
        (z240 is not None and z240 <= -0.5) and
        (rev_yoy is None or rev_yoy > -15)
    )
    turnaround = (
        growth_score >= 8 and
        (rev_yoy is not None and rev_yoy > 0) and
        ((fs.get("roe") is None) or fs.get("roe") < 12 or financial_score < 14)
    )
    moat_reasonable = (
        moat_score >= 10 and financial_score >= 13 and valuation_score >= 6 and
        (z240 is None or z240 < 1.5)
    )

    labels = []
    if wrong_kill:
        labels.append("基本面沒問題被錯殺")
    if turnaround:
        labels.append("轉機股")
    if moat_reasonable:
        labels.append("穩定護城河且股價合理")
    if not labels:
        labels.append("暫不符合三大偏好，列入觀察")

    return {
        "close": close,
        "z240": z240,
        "pe": pe,
        "pbr": pbr,
        "div_yield": div_yield,
        "pe_rank": pe_rank,
        "pbr_rank": pbr_rank,
        "rev_yoy": rev_yoy,
        "rev_3m": rev_3m,
        "inst_5": inst_5,
        "inst_20": inst_20,
        "scores": {
            "產業賽道": industry_score,
            "護城河": moat_score,
            "財務體質": financial_score,
            "成長性": growth_score,
            "估值便宜度": valuation_score,
            "籌碼面": chip_score,
            "技術位置": tech_score,
            "治理風險": governance_score,
            "總分": total,
        },
        "labels": labels,
    }


def plot_bollinger(price: pd.DataFrame, window: int = 240) -> go.Figure:
    df = price.dropna(subset=["close"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="收盤價"))
    for col, name in [(f"MA{window}", f"MA{window}"), (f"U1_{window}", "+1SD"), (f"L1_{window}", "-1SD"), (f"U2_{window}", "+2SD"), (f"L2_{window}", "-2SD")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name))
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20), title=f"{window}日布林通道")
    return fig


def plot_month_revenue(rev: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if rev.empty:
        return fig
    rcol = "revenue" if "revenue" in rev.columns else None
    if rcol:
        fig.add_trace(go.Bar(x=rev["date"], y=rev[rcol], name="月營收"))
    if "revenue_yoy" in rev.columns:
        fig.add_trace(go.Scatter(x=rev["date"], y=rev["revenue_yoy"], name="YoY %", yaxis="y2"))
    fig.update_layout(height=380, title="月營收與年增率", yaxis2=dict(overlaying="y", side="right", ticksuffix="%"), margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_inst(inst: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if inst.empty:
        return fig
    for col in inst.columns:
        if col != "date":
            fig.add_trace(go.Bar(x=inst["date"], y=inst[col], name=col))
    fig.update_layout(height=360, title="三大法人買賣超", barmode="relative", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def rule_based_industry_text(company: Dict[str, Any], result: Dict[str, Any], fs: Dict[str, Any]) -> str:
    industry = company.get("industry_category", "未知產業") or "未知產業"
    name = company.get("stock_name", "") or company.get("stock_id", "")
    labels = "、".join(result["labels"])
    moat_score = result["scores"]["護城河"]
    return f"""
### 產業與護城河初步判讀（規則版）

**{name}** 的產業分類為 **{industry}**。目前系統分類為：**{labels}**。

- 護城河分數：**{moat_score}/15**。這是依產業屬性、ROE、毛利率與財務穩定度做的初步評分。
- 若是轉機股，要優先確認：月營收是否連續改善、毛利率是否跟著改善、現金流是否能撐到轉機落地。
- 若是錯殺股，要確認：本業沒有壞掉、負債沒有惡化、PB/PE 是否真的低於歷史區間。
- 若是護城河股，要避免在 240 日 +2SD 附近追高；好公司買太貴，心情會像買了豪宅卻每天漏水。

**市佔率提醒：** 多數台股中小型公司不會直接揭露市佔率。本系統第一版會先列出產業分類與同業比較邏輯；若要精準市佔率，後續可加入年報/法說會 PDF 擷取與自訂同業資料表。
"""


def ai_industry_analysis(company: Dict[str, Any], result: Dict[str, Any], fs: Dict[str, Any], latest_rev: Dict[str, Any]) -> Optional[str]:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    model = get_secret("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    prompt = f"""
你是台股基本面與產業分析師。請根據下列資料，用繁體中文輸出手機易讀的分析。
不要假裝知道沒有提供的市佔率；不知道就說需要年報/法說會佐證。

公司資料：{json.dumps(company, ensure_ascii=False)}
量化結果：{json.dumps(result, ensure_ascii=False, default=str)}
財務摘要：{json.dumps(fs, ensure_ascii=False, default=str)}
最新月營收：{json.dumps(latest_rev, ensure_ascii=False, default=str)}

請輸出：
1. 行業賽道判斷
2. 護城河來源
3. 可能競爭對手如何找
4. 市佔率可否判斷
5. 三類股票分類理由
6. 主要風險與追蹤清單
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI 分析暫時失敗：{e}"


# -----------------------------
# UI
# -----------------------------

st.title("📊 台股錯殺・轉機・護城河估值系統")
st.caption("輸入台股代號，自動整理股價位置、估值、財務、月營收、法人籌碼，並判斷是否接近你的三種偏好股票。")

with st.sidebar:
    st.header("設定")
    stock_id = normalize_stock_id(st.text_input("股票代號", value="6969", placeholder="例如 6969、8341、1229"))
    boll_window = st.selectbox("布林通道", [240, 60, 20], index=0)
    show_raw = st.toggle("顯示原始資料", value=False)
    st.info("FinMind 可免 Key 使用；若常部署使用，建議在 Streamlit Secrets 加 FINMIND_TOKEN。")

if not stock_id:
    st.stop()

if st.button("開始分析", type="primary"):
    st.session_state["run"] = True

if not st.session_state.get("run", False):
    st.write("按左側或下方按鈕開始。")
    st.stop()

with st.spinner("讀取台股資料並計算估值模型..."):
    company = get_company_row(stock_id)
    price = get_price(stock_id)
    per = get_per_pbr(stock_id)
    rev = get_month_revenue(stock_id)
    inst_raw = get_institutional(stock_id)
    inst = pivot_inst(inst_raw)
    margin = get_margin(stock_id)
    fin = get_financials(stock_id)
    fs = financial_summary(fin)
    result = score_and_classify(company, price, per, rev, inst, fs)

name = company.get("stock_name", "") or "未知公司"
industry = company.get("industry_category", "") or "未知產業"
market_type = company.get("type", "") or "—"

st.subheader(f"{stock_id} {name}")
st.write(f"市場/類型：**{market_type}** ｜ 產業分類：**{industry}**")

# top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新收盤價", fmt_num(result["close"], 2))
c2.metric("PE", fmt_num(result["pe"], 2))
c3.metric("PB", fmt_num(result["pbr"], 2))
c4.metric("240日Z分數", fmt_num(result["z240"], 2))

c5, c6, c7, c8 = st.columns(4)
c5.metric("ROE", fmt_num(fs.get("roe"), 2, "%"))
c6.metric("負債比", fmt_num(fs.get("debt_ratio"), 2, "%"))
c7.metric("最新月營收YoY", fmt_num(result["rev_yoy"], 2, "%"))
c8.metric("法人20日合計", fmt_int(result["inst_20"]))

st.markdown("### 🧭 三類股票判斷")
st.success("、".join(result["labels"]))

scores = result["scores"]
score_df = pd.DataFrame([{"項目": k, "分數": v} for k, v in scores.items() if k != "總分"])
st.metric("總分", f"{scores['總分']} / 100")
st.dataframe(score_df, use_container_width=True, hide_index=True)

# Tabs
tabs = st.tabs(["估值與技術", "月營收", "法人籌碼", "財務體質", "Google Trends", "AI/質化分析", "追蹤清單"])

with tabs[0]:
    if price.empty:
        st.warning("找不到股價資料。")
    else:
        st.plotly_chart(plot_bollinger(price, boll_window), use_container_width=True)
        latest = latest_row(price)
        st.write(f"目前相對 {boll_window} 日均線 Z 分數：**{fmt_num(latest.get(f'Z_{boll_window}'), 2)}**")
        st.caption("Z < -2 常代表極度悲觀，但仍需確認基本面沒有壞掉；Z > +2 可能是過熱，也可能是趨勢爆發。")
    if not per.empty:
        st.markdown("#### PE/PB 歷史分位")
        st.write(f"PE 分位：**{fmt_num(result['pe_rank'], 1, '%')}** ｜ PB 分位：**{fmt_num(result['pbr_rank'], 1, '%')}**")
        st.dataframe(per.tail(10), use_container_width=True, hide_index=True)

with tabs[1]:
    if rev.empty:
        st.warning("找不到月營收資料。")
    else:
        st.plotly_chart(plot_month_revenue(rev), use_container_width=True)
        show_cols = [c for c in ["date", "revenue", "revenue_mom", "revenue_yoy", "revenue_3m_yoy_avg"] if c in rev.columns]
        st.dataframe(rev[show_cols].tail(18), use_container_width=True, hide_index=True)

with tabs[2]:
    if inst.empty:
        st.warning("找不到三大法人資料。")
    else:
        st.plotly_chart(plot_inst(inst.tail(60)), use_container_width=True)
        st.write(f"5日合計：**{fmt_int(result['inst_5'])}** ｜ 20日合計：**{fmt_int(result['inst_20'])}**")
        st.dataframe(inst.tail(30), use_container_width=True, hide_index=True)
    if not margin.empty:
        st.markdown("#### 融資融券")
        st.dataframe(margin.tail(20), use_container_width=True, hide_index=True)

with tabs[3]:
    fin_df = pd.DataFrame([
        {"指標": "營收", "數值": fmt_int(fs.get("revenue"))},
        {"指標": "稅後淨利", "數值": fmt_int(fs.get("net_income"))},
        {"指標": "EPS", "數值": fmt_num(fs.get("eps"), 2)},
        {"指標": "ROE", "數值": fmt_num(fs.get("roe"), 2, "%")},
        {"指標": "ROA", "數值": fmt_num(fs.get("roa"), 2, "%")},
        {"指標": "毛利率", "數值": fmt_num(fs.get("gross_margin"), 2, "%")},
        {"指標": "營益率", "數值": fmt_num(fs.get("op_margin"), 2, "%")},
        {"指標": "負債比", "數值": fmt_num(fs.get("debt_ratio"), 2, "%")},
        {"指標": "營業現金流", "數值": fmt_int(fs.get("ocf"))},
        {"指標": "自由現金流估算", "數值": fmt_int(fs.get("fcf"))},
    ])
    st.dataframe(fin_df, use_container_width=True, hide_index=True)
    st.caption("財報欄位由公開資料轉換，部分公司欄位名稱可能不同；若顯示 —，代表需要人工核對財報科目。")

with tabs[4]:
    st.markdown("### 🔎 Google Trends：台灣 / 美國搜尋熱度")
    st.caption("這裡用 Google Trends 相對搜尋熱度，不是實際搜尋次數。100 代表所選地區與期間內的最高熱度。")
    st.info("Google Trends 適合看大眾關注度、題材熱度與品牌/產品聲量；不代表營收，也不等於股價會漲。")

    trend_mode = st.radio("模式", ["熱門搜尋榜", "關鍵字比較", "相關搜尋"], horizontal=True)

    if trend_mode == "熱門搜尋榜":
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("#### 台灣熱門搜尋")
            tw_hot, tw_err = trends_hot_searches("台灣")
            if tw_err:
                st.warning(tw_err)
            if not tw_hot.empty:
                st.dataframe(tw_hot.head(25), use_container_width=True, hide_index=True)
        with tc2:
            st.markdown("#### 美國熱門搜尋")
            us_hot, us_err = trends_hot_searches("美國")
            if us_err:
                st.warning(us_err)
            if not us_hot.empty:
                st.dataframe(us_hot.head(25), use_container_width=True, hide_index=True)

    elif trend_mode == "關鍵字比較":
        default_keywords = ", ".join([x for x in [name, stock_id, industry] if x and x != "未知產業"][:3])
        raw_keywords = st.text_area("輸入最多 5 個關鍵字，用逗號或換行分隔", value=default_keywords or "台積電, AI伺服器, 半導體")
        keywords = clean_keywords(raw_keywords)
        timeframe_label = st.selectbox("時間範圍", ["過去 7 天", "過去 30 天", "過去 3 個月", "過去 12 個月", "過去 5 年"], index=3)
        timeframe_map = {
            "過去 7 天": "now 7-d",
            "過去 30 天": "today 1-m",
            "過去 3 個月": "today 3-m",
            "過去 12 個月": "today 12-m",
            "過去 5 年": "today 5-y",
        }
        timeframe = timeframe_map[timeframe_label]
        if keywords:
            tw_cfg = GOOGLE_TRENDS_REGION_MAP["台灣"]
            us_cfg = GOOGLE_TRENDS_REGION_MAP["美國"]
            tw_df, tw_err = trends_interest_over_time(tuple(keywords), tw_cfg["geo"], timeframe, tw_cfg["hl"], tw_cfg["tz"])
            us_df, us_err = trends_interest_over_time(tuple(keywords), us_cfg["geo"], timeframe, us_cfg["hl"], us_cfg["tz"])
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("#### 台灣搜尋熱度")
                if tw_err:
                    st.warning(tw_err)
                if not tw_df.empty:
                    st.plotly_chart(plot_trends_interest(tw_df, "台灣 Google Trends"), use_container_width=True)
                    st.dataframe(tw_df.tail(12), use_container_width=True, hide_index=True)
            with tc2:
                st.markdown("#### 美國搜尋熱度")
                if us_err:
                    st.warning(us_err)
                if not us_df.empty:
                    st.plotly_chart(plot_trends_interest(us_df, "美國 Google Trends"), use_container_width=True)
                    st.dataframe(us_df.tail(12), use_container_width=True, hide_index=True)
            st.caption("提醒：中文關鍵字在美國可能搜尋量偏低；美國市場建議輸入英文，例如 TSMC、Nvidia、AI server、semiconductor。")
        else:
            st.warning("請輸入至少一個關鍵字。")

    else:
        default_keyword = name if name and name != "未知公司" else "台積電"
        keyword = st.text_input("關鍵字", value=default_keyword)
        region_label = st.selectbox("地區", ["台灣", "美國"], index=0)
        timeframe = st.selectbox("相關搜尋期間", ["today 12-m", "today 3-m", "today 1-m", "now 7-d", "today 5-y"], index=0)
        cfg = GOOGLE_TRENDS_REGION_MAP[region_label]
        if keyword:
            rq, err = trends_related_queries(keyword, cfg["geo"], timeframe, cfg["hl"], cfg["tz"])
            if err:
                st.warning(err)
            data = rq.get(keyword, {}) if isinstance(rq, dict) else {}
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("#### Top 相關搜尋")
                top = data.get("top") if isinstance(data, dict) else None
                if isinstance(top, pd.DataFrame) and not top.empty:
                    st.dataframe(top, use_container_width=True, hide_index=True)
                else:
                    st.write("沒有資料。")
            with rc2:
                st.markdown("#### Rising 飆升搜尋")
                rising = data.get("rising") if isinstance(data, dict) else None
                if isinstance(rising, pd.DataFrame) and not rising.empty:
                    st.dataframe(rising, use_container_width=True, hide_index=True)
                else:
                    st.write("沒有資料。")
        st.caption("Rising 若顯示 breakout，代表相對成長非常高；但基期可能很低，要搭配新聞與成交量判讀。")

with tabs[5]:
    ai_text = ai_industry_analysis(company, result, fs, latest_row(rev))
    if ai_text:
        st.markdown(ai_text)
    else:
        st.markdown(rule_based_industry_text(company, result, fs))
        st.info("若在 Streamlit Secrets 加入 OPENAI_API_KEY，這裡會升級成 AI 產業/護城河/競爭對手分析。")

with tabs[6]:
    st.markdown("### 轉機/錯殺/護城河追蹤清單")
    checklist = [
        "月營收 YoY 是否連續 3 個月改善",
        "毛利率與營益率是否同步改善",
        "ROE 是否維持或恢復到 8% / 12% 以上",
        "營業現金流是否為正，自由現金流是否惡化",
        "PB/PE 是否低於自身歷史均值或低分位",
        "股價是否落在 240日 -1SD 至 -2SD 的便宜區，而非基本面崩壞區",
        "法人是否由連賣轉為買超，投信是否開始建倉",
        "融資是否過熱；股價跌但融資大增要小心",
        "是否有明確催化劑：新廠、許可、客戶驗證、訂單、政策鬆綁、轉上市",
        "是否有治理風險：董監高質押、頻繁增資、CB稀釋、重大訴訟",
    ]
    for item in checklist:
        st.checkbox(item, value=False)

if show_raw:
    st.markdown("---")
    st.markdown("### 原始資料")
    with st.expander("price"):
        st.dataframe(price, use_container_width=True)
    with st.expander("per/pbr"):
        st.dataframe(per, use_container_width=True)
    with st.expander("month revenue"):
        st.dataframe(rev, use_container_width=True)
    with st.expander("institutional raw"):
        st.dataframe(inst_raw, use_container_width=True)
    with st.expander("financial raw"):
        st.json({k: v.tail(20).to_dict(orient="records") if isinstance(v, pd.DataFrame) and not v.empty else [] for k, v in fin.items()})

st.markdown("---")
st.caption("本工具用於研究輔助，不是買賣建議。台股中小型公司的市佔率、客戶、產能與許可常需要年報、法說會、重大訊息人工佐證。")
