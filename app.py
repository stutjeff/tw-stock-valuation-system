import os
import json
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "全球錯殺・轉機・護城河估值系統"
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

# -----------------------------
# Common helpers
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
    return "".join(ch for ch in s.strip() if ch.isalnum() or ch in [".", "-"]).upper()


def years_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def to_numeric_cols(df: pd.DataFrame, skip: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if col not in skip:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
        out[f"Z_{w}"] = (out["close"] - ma) / sd.replace(0, np.nan)
    return out


def latest_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def percentile_rank(series: pd.Series, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 20:
        return None
    return (s <= value).mean() * 100


def bool_icon(v: Optional[bool]) -> str:
    if v is True:
        return "✅"
    if v is False:
        return "❌"
    return "—"


def evidence_bool(v: Optional[bool], evidence: str) -> str:
    return f"{bool_icon(v)} {evidence}"

# -----------------------------
# Taiwan / FinMind
# -----------------------------

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
        r = requests.get(FINMIND_URL, params=params, timeout=25)
        r.raise_for_status()
        payload = r.json()
        return pd.DataFrame(payload.get("data", []))
    except Exception as e:
        st.warning(f"FinMind 讀取 {dataset} 失敗：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def stock_info() -> pd.DataFrame:
    return finmind("TaiwanStockInfo")


def get_company_row(stock_id: str) -> Dict[str, Any]:
    df = stock_info()
    if df.empty:
        return {"stock_id": stock_id, "stock_name": "", "industry_category": "", "type": ""}
    row = df[df["stock_id"].astype(str) == str(stock_id)]
    if row.empty:
        return {"stock_id": stock_id, "stock_name": "", "industry_category": "", "type": ""}
    return row.iloc[0].to_dict()


def get_tw_price(stock_id: str, days: int = 900) -> pd.DataFrame:
    df = finmind("TaiwanStockPrice", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = to_numeric_cols(df, ["date", "stock_id"])
    df = df.sort_values("date")
    return add_bollinger(df)


def get_tw_per_pbr(stock_id: str, days: int = 900) -> pd.DataFrame:
    df = finmind("TaiwanStockPER", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = to_numeric_cols(df, ["date", "stock_id"])
    return df.sort_values("date")


def get_tw_month_revenue(stock_id: str, months_back: int = 48) -> pd.DataFrame:
    df = finmind("TaiwanStockMonthRevenue", stock_id, years_ago(months_back * 31), date.today().isoformat())
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = to_numeric_cols(df, ["date", "stock_id", "country"])
    df = df.sort_values("date")
    if "revenue" in df.columns:
        df["revenue_yoy"] = df["revenue"].pct_change(12) * 100
        df["revenue_mom"] = df["revenue"].pct_change(1) * 100
        df["revenue_3m_yoy_avg"] = df["revenue_yoy"].rolling(3).mean()
        df["revenue_yoy_improve_3m"] = df["revenue_yoy"].diff().rolling(3).apply(lambda x: 1 if np.all(x > 0) else 0, raw=True)
    return df


def get_tw_institutional(stock_id: str, days: int = 90) -> pd.DataFrame:
    df = finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = to_numeric_cols(df, ["date", "stock_id", "name"])
    return df.sort_values("date")


def pivot_inst(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "name" not in df.columns:
        return pd.DataFrame()
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


def get_tw_margin(stock_id: str, days: int = 180) -> pd.DataFrame:
    df = finmind("TaiwanStockMarginPurchaseShortSale", stock_id, years_ago(days), date.today().isoformat())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = to_numeric_cols(df, ["date", "stock_id"])
    return df.sort_values("date")


def get_tw_financials(stock_id: str, days: int = 1700) -> Dict[str, pd.DataFrame]:
    start = years_ago(days)
    end = date.today().isoformat()
    return {
        "income": finmind("TaiwanStockFinancialStatements", stock_id, start, end),
        "balance": finmind("TaiwanStockBalanceSheet", stock_id, start, end),
        "cashflow": finmind("TaiwanStockCashFlowsStatement", stock_id, start, end),
    }


def tidy_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    out = to_numeric_cols(out, ["date", "stock_id", "type", "origin_name"])
    return out.sort_values("date") if "date" in out.columns else out


def extract_metric(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    if df.empty:
        return None
    name_cols = [c for c in ["type", "origin_name"] if c in df.columns]
    if not name_cols or "value" not in df.columns:
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


def tw_financial_summary(fin: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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
        fcf = ocf + capex if capex is not None and capex < 0 else ocf - abs(capex or 0)

    return {"revenue": revenue, "net_income": net_income, "eps": eps, "roe": roe, "roa": roa, "debt_ratio": debt_ratio, "gross_margin": gross_margin, "op_margin": op_margin, "ocf": ocf, "fcf": fcf}


def tw_score_and_classify(company: Dict[str, Any], price: pd.DataFrame, per: pd.DataFrame, rev: pd.DataFrame, inst_p: pd.DataFrame, fs: Dict[str, Any]) -> Dict[str, Any]:
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
        inst_5 = safe_float(inst_p.tail(5)["合計"].sum())
        inst_20 = safe_float(inst_p.tail(20)["合計"].sum())

    industry_score = 8
    industry = str(company.get("industry_category", ""))
    moat_keywords = ["半導體", "電機", "電子", "通信", "水泥", "食品", "金融", "生技", "環保", "綠能", "其他電子"]
    moat_score = 7 if any(k in industry for k in moat_keywords) else 5
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
    tech_score = 5 if z240 is not None and -2 <= z240 <= 0.5 else 3 if z240 is not None and z240 < -2 else 1 if z240 is not None and z240 > 2 else 2
    governance_score = 3
    total = industry_score + moat_score + financial_score + growth_score + valuation_score + chip_score + tech_score + governance_score

    labels = []
    wrong_kill = financial_score >= 12 and valuation_score >= 8 and z240 is not None and z240 <= -0.5 and (rev_yoy is None or rev_yoy > -15)
    turnaround = growth_score >= 8 and rev_yoy is not None and rev_yoy > 0 and (fs.get("roe") is None or fs.get("roe") < 12 or financial_score < 14)
    moat_reasonable = moat_score >= 10 and financial_score >= 13 and valuation_score >= 6 and (z240 is None or z240 < 1.5)
    if wrong_kill:
        labels.append("基本面沒問題被錯殺")
    if turnaround:
        labels.append("轉機股")
    if moat_reasonable:
        labels.append("穩定護城河且股價合理")
    if not labels:
        labels.append("暫不符合三大偏好，列入觀察")

    return {"market": "TW", "close": close, "z240": z240, "pe": pe, "pbr": pbr, "div_yield": div_yield, "pe_rank": pe_rank, "pbr_rank": pbr_rank, "rev_yoy": rev_yoy, "rev_3m": rev_3m, "inst_5": inst_5, "inst_20": inst_20, "scores": {"產業賽道": industry_score, "護城河": moat_score, "財務體質": financial_score, "成長性": growth_score, "估值便宜度": valuation_score, "籌碼面": chip_score, "技術位置": tech_score, "治理風險": governance_score, "總分": total}, "labels": labels}

# -----------------------------
# US / Japan via yfinance
# -----------------------------

def normalize_global_ticker(raw: str, market: str) -> str:
    s = normalize_stock_id(raw)
    if market == "日股" and s and "." not in s:
        return f"{s}.T"
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def yf_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
        if df.empty:
            return df
        df = df.reset_index()
        # Flatten possible MultiIndex columns
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        rename_map = {"date": "date", "open": "open", "high": "max", "low": "min", "close": "close", "volume": "volume"}
        df = df.rename(columns=rename_map)
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return add_bollinger(df.sort_values("date"))
    except Exception as e:
        st.warning(f"yfinance 股價讀取失敗：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def yf_info(ticker: str) -> Dict[str, Any]:
    if yf is None:
        return {}
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        return info
    except Exception as e:
        st.warning(f"yfinance 公司資料讀取失敗：{e}")
        return {}


def pct_from_ratio(x: Any) -> Optional[float]:
    v = safe_float(x)
    if v is None:
        return None
    return v * 100 if abs(v) <= 5 else v


def global_summary(ticker: str, market: str, price: pd.DataFrame, info: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latest = latest_row(price)
    close = safe_float(latest.get("close")) or safe_float(info.get("currentPrice")) or safe_float(info.get("regularMarketPrice"))
    z240 = safe_float(latest.get("Z_240"))
    pe = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))
    pbr = safe_float(info.get("priceToBook"))
    roe = pct_from_ratio(info.get("returnOnEquity"))
    roa = pct_from_ratio(info.get("returnOnAssets"))
    gross_margin = pct_from_ratio(info.get("grossMargins"))
    op_margin = pct_from_ratio(info.get("operatingMargins"))
    revenue_growth = pct_from_ratio(info.get("revenueGrowth"))
    earnings_growth = pct_from_ratio(info.get("earningsGrowth"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    ocf = safe_float(info.get("operatingCashflow"))
    fcf = safe_float(info.get("freeCashflow"))
    sector = info.get("sector") or "—"
    industry = info.get("industry") or "—"
    name = info.get("longName") or info.get("shortName") or ticker
    market_cap = safe_float(info.get("marketCap"))
    dividend_yield = pct_from_ratio(info.get("dividendYield"))

    fs = {"revenue": safe_float(info.get("totalRevenue")), "net_income": safe_float(info.get("netIncomeToCommon")), "eps": safe_float(info.get("trailingEps")), "roe": roe, "roa": roa, "debt_ratio": None, "debt_to_equity": debt_to_equity, "gross_margin": gross_margin, "op_margin": op_margin, "ocf": ocf, "fcf": fcf, "revenue_growth": revenue_growth, "earnings_growth": earnings_growth}

    financial_score = 0
    if roe is not None:
        financial_score += 8 if roe >= 15 else 6 if roe >= 10 else 3 if roe > 0 else 0
    if op_margin is not None:
        financial_score += 5 if op_margin >= 15 else 3 if op_margin > 0 else 0
    if fcf is not None:
        financial_score += 4 if fcf > 0 else 1
    if debt_to_equity is not None:
        financial_score += 3 if debt_to_equity < 80 else 2 if debt_to_equity < 150 else 0
    financial_score = min(20, financial_score)

    growth_score = 0
    if revenue_growth is not None:
        growth_score += 8 if revenue_growth >= 20 else 6 if revenue_growth >= 10 else 4 if revenue_growth >= 0 else 1
    if earnings_growth is not None:
        growth_score += 7 if earnings_growth >= 20 else 5 if earnings_growth >= 5 else 2 if earnings_growth >= 0 else 0
    growth_score = min(15, growth_score)

    valuation_score = 0
    if pe is not None and pe > 0:
        valuation_score += 6 if pe <= 15 else 4 if pe <= 25 else 2 if pe <= 40 else 0
    if pbr is not None:
        valuation_score += 5 if pbr <= 2 else 3 if pbr <= 5 else 1 if pbr <= 10 else 0
    if z240 is not None:
        valuation_score += 4 if z240 <= -1 else 2 if z240 <= 0 else 0
    valuation_score = min(15, valuation_score)

    moat_score = 5
    if roe is not None and roe >= 15:
        moat_score += 4
    if gross_margin is not None and gross_margin >= 35:
        moat_score += 3
    if market_cap is not None and market_cap >= 10_000_000_000:
        moat_score += 2
    moat_score = min(15, moat_score)

    industry_score = 8
    chip_score = 0  # US/JP 不做三大法人；保留欄位但不亂判斷
    tech_score = 5 if z240 is not None and -2 <= z240 <= 0.5 else 3 if z240 is not None and z240 < -2 else 1 if z240 is not None and z240 > 2 else 2
    governance_score = 3
    total = industry_score + moat_score + financial_score + growth_score + valuation_score + chip_score + tech_score + governance_score

    wrong_kill = financial_score >= 13 and valuation_score >= 8 and z240 is not None and z240 <= -0.5 and (revenue_growth is None or revenue_growth > -10)
    turnaround = growth_score >= 8 and (revenue_growth is not None and revenue_growth > 0) and financial_score < 15
    moat_reasonable = moat_score >= 10 and financial_score >= 13 and valuation_score >= 5 and (z240 is None or z240 < 1.5)

    labels = []
    if wrong_kill:
        labels.append("基本面沒問題被錯殺")
    if turnaround:
        labels.append("轉機股")
    if moat_reasonable:
        labels.append("穩定護城河且股價合理")
    if not labels:
        labels.append("暫不符合三大偏好，列入觀察")

    company = {"stock_id": ticker, "stock_name": name, "industry_category": f"{sector} / {industry}", "type": market}
    result = {"market": market, "close": close, "z240": z240, "pe": pe, "pbr": pbr, "div_yield": dividend_yield, "pe_rank": None, "pbr_rank": None, "rev_yoy": revenue_growth, "rev_3m": None, "inst_5": None, "inst_20": None, "scores": {"產業賽道": industry_score, "護城河": moat_score, "財務體質": financial_score, "成長性": growth_score, "估值便宜度": valuation_score, "籌碼面": chip_score, "技術位置": tech_score, "治理風險": governance_score, "總分": total}, "labels": labels, "market_cap": market_cap, "sector": sector, "industry": industry}
    return company, {"fs": fs, "result": result}

# -----------------------------
# Auto checklist
# -----------------------------

def build_auto_checklist(market: str, result: Dict[str, Any], fs: Dict[str, Any], rev: pd.DataFrame = pd.DataFrame(), inst: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    z240 = safe_float(result.get("z240"))
    pe = safe_float(result.get("pe"))
    pbr = safe_float(result.get("pbr"))
    pe_rank = safe_float(result.get("pe_rank"))
    pbr_rank = safe_float(result.get("pbr_rank"))
    rev_yoy = safe_float(result.get("rev_yoy"))
    roe = safe_float(fs.get("roe"))
    gross_margin = safe_float(fs.get("gross_margin"))
    op_margin = safe_float(fs.get("op_margin"))
    ocf = safe_float(fs.get("ocf"))
    fcf = safe_float(fs.get("fcf"))
    debt_ratio = safe_float(fs.get("debt_ratio"))
    debt_to_equity = safe_float(fs.get("debt_to_equity"))

    # Revenue improvement
    rev_improve = None
    rev_evidence = "資料不足"
    if market == "台股" and not rev.empty and "revenue_yoy" in rev.columns and len(rev.dropna(subset=["revenue_yoy"])) >= 4:
        last4 = rev.dropna(subset=["revenue_yoy"]).tail(4)["revenue_yoy"].tolist()
        rev_improve = last4[-1] > last4[-2] > last4[-3] > last4[-4]
        rev_evidence = f"近4期YoY：{', '.join(fmt_num(x,1,'%') for x in last4)}"
    elif rev_yoy is not None:
        rev_improve = rev_yoy > 0
        rev_evidence = f"營收成長率：{fmt_num(rev_yoy,1,'%')}"

    margin_ok = None
    margin_evidence = "資料不足"
    if gross_margin is not None or op_margin is not None:
        margin_ok = (gross_margin is None or gross_margin > 15) and (op_margin is None or op_margin > 5)
        margin_evidence = f"毛利率 {fmt_num(gross_margin,1,'%')} / 營益率 {fmt_num(op_margin,1,'%')}"

    roe_ok = None if roe is None else roe >= 8
    roe_evidence = f"ROE {fmt_num(roe,1,'%')}"

    cash_ok = None
    cash_evidence = "資料不足"
    if ocf is not None or fcf is not None:
        cash_ok = (ocf is None or ocf > 0) and (fcf is None or fcf > 0)
        cash_evidence = f"OCF {fmt_int(ocf)} / FCF {fmt_int(fcf)}"

    valuation_low = None
    if pbr_rank is not None or pe_rank is not None:
        valuation_low = (pbr_rank is not None and pbr_rank <= 40) or (pe_rank is not None and pe_rank <= 40)
        valuation_evidence = f"PE分位 {fmt_num(pe_rank,1,'%')} / PB分位 {fmt_num(pbr_rank,1,'%')}"
    elif pe is not None or pbr is not None:
        valuation_low = (pe is not None and 0 < pe <= 20) or (pbr is not None and pbr <= 2.5)
        valuation_evidence = f"PE {fmt_num(pe,1)} / PB {fmt_num(pbr,1)}"
    else:
        valuation_evidence = "資料不足"

    price_bargain = None if z240 is None else (-2.5 <= z240 <= -0.8)
    price_evidence = f"240日Z分數 {fmt_num(z240,2)}"

    inst_turn = None
    inst_evidence = "台股才有三大法人；美股/日股此項不判斷"
    if market == "台股" and not inst.empty and "合計" in inst.columns and len(inst) >= 20:
        last5 = safe_float(inst.tail(5)["合計"].sum())
        prev15 = safe_float(inst.tail(20).head(15)["合計"].sum())
        inst_turn = (last5 is not None and last5 > 0) and (prev15 is None or prev15 <= 0 or last5 > abs(prev15) * 0.2)
        inst_evidence = f"近5日 {fmt_int(last5)} / 前15日 {fmt_int(prev15)}"

    financing_ok = None
    financing_evidence = "融資過熱需搭配融資融券資料；此版先不自動判斷"

    catalyst = None
    catalyst_evidence = "新廠、許可、訂單、客戶驗證、政策鬆綁仍需人工確認"

    governance = None
    if debt_ratio is not None:
        governance = debt_ratio < 70
        governance_evidence = f"負債比 {fmt_num(debt_ratio,1,'%')}；董監質押/CB/訴訟仍需人工確認"
    elif debt_to_equity is not None:
        governance = debt_to_equity < 150
        governance_evidence = f"Debt/Equity {fmt_num(debt_to_equity,1)}；治理風險仍需人工確認"
    else:
        governance_evidence = "資料不足；需查董監質押、增資、CB、訴訟"

    rows = [
        ("轉機", "月營收 / 營收成長是否改善", rev_improve, rev_evidence),
        ("轉機/護城河", "毛利率與營益率是否健康", margin_ok, margin_evidence),
        ("護城河", "ROE 是否維持或恢復到 8% / 12% 以上", roe_ok, roe_evidence),
        ("錯殺/護城河", "營業現金流與自由現金流是否為正", cash_ok, cash_evidence),
        ("錯殺", "PB/PE 是否低於自身歷史均值或合理低檔", valuation_low, valuation_evidence),
        ("錯殺", "股價是否落在 240日 -1SD 至 -2SD 附近", price_bargain, price_evidence),
        ("籌碼", "法人是否由連賣轉為買超 / 投信是否建倉", inst_turn, inst_evidence),
        ("風險", "融資是否過熱", financing_ok, financing_evidence),
        ("轉機", "是否有明確催化劑", catalyst, catalyst_evidence),
        ("風險", "是否沒有明顯治理風險", governance, governance_evidence),
    ]
    return pd.DataFrame([{"類型": a, "條件": b, "系統判斷": bool_icon(c), "自動勾選": c is True, "證據": d} for a, b, c, d in rows])


def show_auto_checkboxes(check_df: pd.DataFrame):
    st.markdown("### 系統自動勾選清單")
    st.caption("✅ 代表系統依目前資料判斷條件成立；❌ 代表不成立；— 代表資料不足或仍需人工確認。勾選為自動判斷，不能手動改。")
    for _, row in check_df.iterrows():
        st.checkbox(f"{row['類型']}｜{row['條件']} — {row['證據']}", value=bool(row["自動勾選"]), disabled=True)
    st.dataframe(check_df[["類型", "條件", "系統判斷", "證據"]], use_container_width=True, hide_index=True)

# -----------------------------
# Charts / text
# -----------------------------

def plot_bollinger(price: pd.DataFrame, window: int = 240) -> go.Figure:
    df = price.dropna(subset=["close"]).copy()
    fig = go.Figure()
    if df.empty:
        return fig
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
    if "revenue" in rev.columns:
        fig.add_trace(go.Bar(x=rev["date"], y=rev["revenue"], name="月營收"))
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
- 若是轉機股，要優先確認：營收是否連續改善、毛利率是否跟著改善、現金流是否能撐到轉機落地。
- 若是錯殺股，要確認：本業沒有壞掉、負債沒有惡化、PB/PE 是否真的低於歷史區間。
- 若是護城河股，要避免在 240 日 +2SD 附近追高；好公司買太貴，心情會像買了豪宅卻每天漏水。

**市佔率提醒：** 多數公司不會直接揭露精準市佔率。本系統先做量化篩選；若要精準市佔率，需要年報、法說會、投資人簡報或產業報告佐證。
"""


def ai_industry_analysis(company: Dict[str, Any], result: Dict[str, Any], fs: Dict[str, Any]) -> Optional[str]:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    model = get_secret("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    prompt = f"""
你是全球股票基本面與產業分析師。請根據下列資料，用繁體中文輸出手機易讀的分析。
不要假裝知道沒有提供的市佔率；不知道就說需要年報/法說會/10-K/有價證券報告佐證。

公司資料：{json.dumps(company, ensure_ascii=False, default=str)}
量化結果：{json.dumps(result, ensure_ascii=False, default=str)}
財務摘要：{json.dumps(fs, ensure_ascii=False, default=str)}

請輸出：
1. 行業賽道判斷
2. 護城河來源
3. 可能競爭對手如何找
4. 市佔率可否判斷
5. 三類股票分類理由
6. 主要風險與追蹤清單
"""
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI 分析暫時失敗：{e}"


def render_summary(company: Dict[str, Any], result: Dict[str, Any], fs: Dict[str, Any]):
    name = company.get("stock_name", "") or "未知公司"
    industry = company.get("industry_category", "") or "未知產業"
    market_type = company.get("type", "") or "—"
    st.subheader(f"{company.get('stock_id')} {name}")
    st.write(f"市場/類型：**{market_type}** ｜ 產業分類：**{industry}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新價格", fmt_num(result.get("close"), 2))
    c2.metric("PE", fmt_num(result.get("pe"), 2))
    c3.metric("PB", fmt_num(result.get("pbr"), 2))
    c4.metric("240日Z分數", fmt_num(result.get("z240"), 2))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ROE", fmt_num(fs.get("roe"), 2, "%"))
    c6.metric("負債/槓桿", fmt_num(fs.get("debt_ratio") if fs.get("debt_ratio") is not None else fs.get("debt_to_equity"), 2, "%" if fs.get("debt_ratio") is not None else ""))
    c7.metric("營收成長", fmt_num(result.get("rev_yoy"), 2, "%"))
    c8.metric("法人20日合計", fmt_int(result.get("inst_20")))
    st.markdown("### 🧭 三類股票判斷")
    st.success("、".join(result["labels"]))
    scores = result["scores"]
    st.metric("總分", f"{scores['總分']} / 100")
    score_df = pd.DataFrame([{"項目": k, "分數": v} for k, v in scores.items() if k != "總分"])
    st.dataframe(score_df, use_container_width=True, hide_index=True)


def render_financial_table(fs: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"指標": "營收", "數值": fmt_int(fs.get("revenue"))},
        {"指標": "稅後淨利", "數值": fmt_int(fs.get("net_income"))},
        {"指標": "EPS", "數值": fmt_num(fs.get("eps"), 2)},
        {"指標": "ROE", "數值": fmt_num(fs.get("roe"), 2, "%")},
        {"指標": "ROA", "數值": fmt_num(fs.get("roa"), 2, "%")},
        {"指標": "毛利率", "數值": fmt_num(fs.get("gross_margin"), 2, "%")},
        {"指標": "營益率", "數值": fmt_num(fs.get("op_margin"), 2, "%")},
        {"指標": "負債比 / D/E", "數值": fmt_num(fs.get("debt_ratio") if fs.get("debt_ratio") is not None else fs.get("debt_to_equity"), 2, "%" if fs.get("debt_ratio") is not None else "")},
        {"指標": "營業現金流", "數值": fmt_int(fs.get("ocf"))},
        {"指標": "自由現金流", "數值": fmt_int(fs.get("fcf"))},
    ])

# -----------------------------
# UI
# -----------------------------

st.title("📊 全球錯殺・轉機・護城河估值系統")
st.caption("支援台股、美股、日股。輸入代號後，自動整理股價位置、估值、財務、營收/籌碼，並由系統自動勾選追蹤條件。")

with st.sidebar:
    st.header("設定")
    market = st.radio("市場", ["台股", "美股", "日股"], horizontal=True)
    default_code = "6969" if market == "台股" else "AAPL" if market == "美股" else "7203"
    raw_code = st.text_input("股票代號", value=default_code, placeholder="台股 6969｜美股 AAPL｜日股 7203 或 7203.T")
    boll_window = st.selectbox("布林通道", [240, 60, 20], index=0)
    show_raw = st.toggle("顯示原始資料", value=False)
    st.info("台股資料源：FinMind；美股/日股資料源：yfinance。若常用台股，建議在 Secrets 加 FINMIND_TOKEN。")

code = normalize_stock_id(raw_code)
if market in ["美股", "日股"]:
    code = normalize_global_ticker(code, market)

if not code:
    st.stop()

if st.button("開始分析", type="primary"):
    st.session_state["run"] = True
    st.session_state["last_market"] = market
    st.session_state["last_code"] = code

# 若切換代號或市場，自動允許重新跑
if st.session_state.get("last_market") != market or st.session_state.get("last_code") != code:
    st.session_state["run"] = False

if not st.session_state.get("run", False):
    st.write("按左側或下方按鈕開始。")
    st.stop()

if market == "台股":
    with st.spinner("讀取台股資料並計算估值模型..."):
        company = get_company_row(code)
        price = get_tw_price(code)
        per = get_tw_per_pbr(code)
        rev = get_tw_month_revenue(code)
        inst_raw = get_tw_institutional(code)
        inst = pivot_inst(inst_raw)
        margin = get_tw_margin(code)
        fin = get_tw_financials(code)
        fs = tw_financial_summary(fin)
        result = tw_score_and_classify(company, price, per, rev, inst, fs)
else:
    with st.spinner(f"讀取{market}資料並計算估值模型..."):
        price = yf_history(code)
        info = yf_info(code)
        company, pack = global_summary(code, market, price, info)
        fs = pack["fs"]
        result = pack["result"]
        per = pd.DataFrame()
        rev = pd.DataFrame()
        inst = pd.DataFrame()
        inst_raw = pd.DataFrame()
        margin = pd.DataFrame()
        fin = {}

render_summary(company, result, fs)
check_df = build_auto_checklist(market, result, fs, rev, inst)

base_tabs = ["估值與技術", "財務/營收", "法人籌碼", "AI/質化分析", "自動追蹤清單"]
tabs = st.tabs(base_tabs)

with tabs[0]:
    if price.empty:
        st.warning("找不到股價資料。")
    else:
        st.plotly_chart(plot_bollinger(price, boll_window), use_container_width=True)
        latest = latest_row(price)
        st.write(f"目前相對 {boll_window} 日均線 Z 分數：**{fmt_num(latest.get(f'Z_{boll_window}'), 2)}**")
        st.caption("Z < -2 常代表極度悲觀，但仍需確認基本面沒有壞掉；Z > +2 可能是過熱，也可能是趨勢爆發。")
    if market == "台股" and not per.empty:
        st.markdown("#### PE/PB 歷史分位")
        st.write(f"PE 分位：**{fmt_num(result.get('pe_rank'), 1, '%')}** ｜ PB 分位：**{fmt_num(result.get('pbr_rank'), 1, '%')}**")
        st.dataframe(per.tail(10), use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown("#### 財務體質")
    st.dataframe(render_financial_table(fs), use_container_width=True, hide_index=True)
    if market == "台股":
        st.markdown("#### 月營收")
        if rev.empty:
            st.warning("找不到月營收資料。")
        else:
            st.plotly_chart(plot_month_revenue(rev), use_container_width=True)
            show_cols = [c for c in ["date", "revenue", "revenue_mom", "revenue_yoy", "revenue_3m_yoy_avg"] if c in rev.columns]
            st.dataframe(rev[show_cols].tail(18), use_container_width=True, hide_index=True)
    else:
        st.caption("美股/日股此版使用 yfinance 的營收成長率與財務摘要；若要更精準，可後續接 10-K / 有價證券報告資料。")

with tabs[2]:
    if market != "台股":
        st.info("美股/日股沒有台股三大法人欄位。此版先不做法人籌碼判斷，避免硬湊假訊號。")
    else:
        if inst.empty:
            st.warning("找不到三大法人資料。")
        else:
            st.plotly_chart(plot_inst(inst.tail(60)), use_container_width=True)
            st.write(f"5日合計：**{fmt_int(result.get('inst_5'))}** ｜ 20日合計：**{fmt_int(result.get('inst_20'))}**")
            st.dataframe(inst.tail(30), use_container_width=True, hide_index=True)
        if not margin.empty:
            st.markdown("#### 融資融券")
            st.dataframe(margin.tail(20), use_container_width=True, hide_index=True)

with tabs[3]:
    ai_text = ai_industry_analysis(company, result, fs)
    if ai_text:
        st.markdown(ai_text)
    else:
        st.markdown(rule_based_industry_text(company, result, fs))
        st.info("若在 Streamlit Secrets 加入 OPENAI_API_KEY，這裡會升級成 AI 產業/護城河/競爭對手分析。")

with tabs[4]:
    show_auto_checkboxes(check_df)
    ok_count = int(check_df["自動勾選"].sum())
    known_count = int((check_df["系統判斷"] != "—").sum())
    st.metric("自動符合條件", f"{ok_count} / {len(check_df)}")
    st.caption("這一頁是系統自動打勾，不是手動備忘。未知項目通常代表需要年報、法說會、重大訊息或治理資料佐證。")

if show_raw:
    st.markdown("---")
    st.markdown("### 原始資料")
    with st.expander("price"):
        st.dataframe(price, use_container_width=True)
    if market == "台股":
        with st.expander("per/pbr"):
            st.dataframe(per, use_container_width=True)
        with st.expander("month revenue"):
            st.dataframe(rev, use_container_width=True)
        with st.expander("institutional raw"):
            st.dataframe(inst_raw, use_container_width=True)
        with st.expander("financial raw"):
            st.json({k: v.tail(20).to_dict(orient="records") if isinstance(v, pd.DataFrame) and not v.empty else [] for k, v in fin.items()})
    else:
        with st.expander("yfinance info"):
            st.json(info)

st.markdown("---")
st.caption("本工具用於研究輔助，不是買賣建議。市佔率、客戶、產能與許可常需要年報、法說會、重大訊息、10-K 或有價證券報告人工佐證。")
