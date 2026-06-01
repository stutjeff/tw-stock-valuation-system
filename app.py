
import re, time, requests
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET

SEC_USER_AGENT = "13F-iPhone-Analyzer/2.0 stutjeff@gmail.com"
REQ_DELAY = 0.15
DATA_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
WEB_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def get_json(url):
    time.sleep(REQ_DELAY)
    headers = DATA_HEADERS if "data.sec.gov" in url else WEB_HEADERS
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def get_text(url):
    time.sleep(REQ_DELAY)
    r = requests.get(url, headers=WEB_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def norm_cik(x):
    d = re.sub(r"\D", "", x or "")
    return d.zfill(10) if d else ""


@st.cache_data(ttl=86400)
def company_list():
    raw = get_json("https://www.sec.gov/files/company_tickers.json")
    return pd.DataFrame([
        {"cik": str(v.get("cik_str", "")).zfill(10), "ticker": v.get("ticker", ""), "title": v.get("title", "")}
        for v in raw.values()
    ])


@st.cache_data(ttl=3600)
def submissions(cik10):
    return get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")


def filings_13f(sub):
    recent = sub.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    if df.empty or "form" not in df:
        return pd.DataFrame()
    df = df[df["form"].isin(["13F-HR", "13F-HR/A"])].copy()
    if df.empty:
        return df
    cols = [c for c in ["accessionNumber", "form", "filingDate", "reportDate", "primaryDocument"] if c in df]
    df = df[cols].sort_values(["reportDate", "filingDate"], ascending=[False, False])
    df = df.drop_duplicates("reportDate", keep="first")
    return df.sort_values("reportDate", ascending=False).reset_index(drop=True)


def archive_base(cik10, acc):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc.replace('-', '')}"


def info_table_url(cik10, acc, primary=""):
    base = archive_base(cik10, acc)
    idx = get_json(base + "/index.json")
    candidates = []
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "")
        low = name.lower()
        if low.endswith(".xml") and "xsl" not in low:
            score = 0
            if "infotable" in low or "info_table" in low or "form13f" in low: score += 10
            if "13f" in low: score += 3
            if "primary" not in low: score += 1
            candidates.append((score, name))
    if candidates:
        candidates.sort(reverse=True)
        return base + "/" + candidates[0][1]
    if primary:
        return base + "/" + primary
    raise RuntimeError("找不到 information table XML")


def tagname(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def child_text(el, name, default=""):
    for c in list(el):
        if tagname(c.tag) == name:
            return (c.text or "").strip()
    return default


def nested_text(el, parent, child, default=""):
    for c in list(el):
        if tagname(c.tag) == parent:
            return child_text(c, child, default)
    return default


def num(x):
    try:
        return float(str(x).replace(",", "").strip()) if str(x).strip() else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def parse_xml(url):
    root = ET.fromstring(get_text(url).encode("utf-8"))
    rows = []
    for info in root.iter():
        if tagname(info.tag) != "infoTable":
            continue
        rows.append({
            "issuer": child_text(info, "nameOfIssuer"),
            "class": child_text(info, "titleOfClass"),
            "cusip": child_text(info, "cusip"),
            "put_call": child_text(info, "putCall"),
            "share_type": nested_text(info, "shrsOrPrnAmt", "sshPrnamtType"),
            "market_value_usd": num(child_text(info, "value")) * 1000,
            "shares": num(nested_text(info, "shrsOrPrnAmt", "sshPrnamt")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby(["issuer", "class", "cusip", "put_call", "share_type"], dropna=False, as_index=False).agg({"market_value_usd":"sum", "shares":"sum"})
    total = df["market_value_usd"].sum()
    df["weight_pct"] = df["market_value_usd"] / total * 100 if total else 0
    return df.sort_values("market_value_usd", ascending=False).reset_index(drop=True)


def period(df, suffix):
    return df[["cusip", "put_call", "share_type", "issuer", "class", "market_value_usd", "shares", "weight_pct"]].rename(columns={
        "issuer": f"issuer_{suffix}", "class": f"class_{suffix}", "market_value_usd": f"value_{suffix}", "shares": f"shares_{suffix}", "weight_pct": f"weight_{suffix}"
    })


def first(*xs):
    for x in xs:
        if pd.notna(x) and str(x).strip(): return x
    return ""


def compare3(q0, q1, q2):
    key = ["cusip", "put_call", "share_type"]
    m = period(q0,"q0").merge(period(q1,"q1"), on=key, how="outer").merge(period(q2,"q2"), on=key, how="outer")
    for c in ["issuer_q0","issuer_q1","issuer_q2","class_q0","class_q1","class_q2"]:
        if c not in m: m[c]=""
    m["issuer"] = m.apply(lambda r: first(r["issuer_q0"], r["issuer_q1"], r["issuer_q2"]), axis=1)
    m["class"] = m.apply(lambda r: first(r["class_q0"], r["class_q1"], r["class_q2"]), axis=1)
    for c in ["value_q0","value_q1","value_q2","shares_q0","shares_q1","shares_q2","weight_q0","weight_q1","weight_q2"]:
        m[c] = m.get(c, 0).fillna(0)
    m["share_change_q0_vs_q1"] = m["shares_q0"] - m["shares_q1"]
    m["share_change_q1_vs_q2"] = m["shares_q1"] - m["shares_q2"]
    m["value_change_q0_vs_q1"] = m["value_q0"] - m["value_q1"]
    m["weight_change_q0_vs_q1"] = m["weight_q0"] - m["weight_q1"]
    def pct(a,b): return None if b == 0 else (a-b)/b*100
    m["share_change_pct_q0_vs_q1"] = m.apply(lambda r: pct(r["shares_q0"], r["shares_q1"]), axis=1)
    def trend(r):
        s0,s1,s2 = r["shares_q0"], r["shares_q1"], r["shares_q2"]
        if s2==0 and s1==0 and s0>0: return "本期新進"
        if s2==0 and s1>0 and s0>s1: return "新進後續加碼"
        if s2>0 and s1>0 and s0==0: return "本期清倉"
        if s0>s1>s2: return "連續加碼"
        if s0<s1<s2 and s0>0: return "連續減碼"
        if s0==s1==s2 and s0>0: return "股數不變"
        if s0>s1 and s1<=s2: return "本期轉加碼"
        if s0<s1 and s1>=s2 and s0>0: return "本期轉減碼"
        return "其他"
    m["trend"] = m.apply(trend, axis=1)
    cols = ["trend","issuer","class","cusip","put_call","share_type","shares_q0","shares_q1","shares_q2","share_change_q0_vs_q1","share_change_pct_q0_vs_q1","share_change_q1_vs_q2","value_q0","value_q1","value_q2","value_change_q0_vs_q1","weight_q0","weight_q1","weight_q2","weight_change_q0_vs_q1"]
    return m[cols].sort_values(["value_q0","value_change_q0_vs_q1"], ascending=False).reset_index(drop=True)


def main():
    st.set_page_config(page_title="13F 三期比較工具", layout="wide")
    st.title("13F 三期比較工具")
    st.caption("查詢 SEC 13F-HR，並比較本期、前一期、前兩期。")
    with st.sidebar:
        presets = [("手動輸入", ""),("Berkshire Hathaway / 巴菲特", "0001067983"),("Scion Asset Management / Michael Burry", "0001649339"),("Pershing Square / Bill Ackman", "0001336528"),("Bridgewater Associates", "0001350694"),("Duquesne Family Office", "0001536411")]
        p = st.selectbox("常用機構", presets, format_func=lambda x:x[0])
        q = st.text_input("輸入 CIK 或公司/基金名稱", value=p[1])
        max_rows = st.slider("顯示筆數", 10, 500, 100, 10)
    cik = ""
    if q:
        d = re.sub(r"\D", "", q)
        if len(d) >= 4:
            cik = norm_cik(q)
        else:
            matches = company_list()[lambda x: x["title"].str.contains(q, case=False, na=False)].head(20)
            if not matches.empty:
                choice = st.selectbox("搜尋結果", matches.to_dict("records"), format_func=lambda r: f'{r["title"]} | {r["ticker"]} | CIK {r["cik"]}')
                cik = choice["cik"]
    if not cik:
        st.info("請輸入 13F 管理人的 CIK，CIK 最準。")
        st.stop()
    try:
        sub = submissions(cik)
        name = sub.get("name", "")
        filings = filings_13f(sub)
    except Exception as e:
        st.error(f"讀取 SEC 失敗：{e}"); st.stop()
    st.subheader(f"{name}｜CIK {cik}")
    if len(filings) < 3:
        st.error("可用 13F 少於三期。")
        st.dataframe(filings, use_container_width=True, hide_index=True)
        st.stop()
    idx = st.selectbox("選擇本期", range(len(filings)-2), format_func=lambda i: f'{filings.loc[i,"filingDate"]}｜Report {filings.loc[i,"reportDate"]}')
    selected = filings.iloc[[idx, idx+1, idx+2]].reset_index(drop=True)
    labels = selected["reportDate"].tolist()
    st.markdown("### 比較期數")
    st.dataframe(selected[["form","filingDate","reportDate","accessionNumber"]], use_container_width=True, hide_index=True)
    try:
        urls=[]; dfs=[]
        for _, row in selected.iterrows():
            url = info_table_url(cik, row["accessionNumber"], row.get("primaryDocument", ""))
            urls.append(url); dfs.append(parse_xml(url))
    except Exception as e:
        st.error(f"解析 13F 失敗：{e}"); st.stop()
    q0,q1,q2 = dfs
    overview = pd.DataFrame({"reportDate":labels,"total_value_usd":[q0.market_value_usd.sum(),q1.market_value_usd.sum(),q2.market_value_usd.sum()],"holding_count":[len(q0),len(q1),len(q2)],"top10_weight_pct":[q0.head(10).weight_pct.sum(),q1.head(10).weight_pct.sum(),q2.head(10).weight_pct.sum()]})
    st.markdown("### 三期總覽")
    st.dataframe(overview.style.format({"total_value_usd":"${:,.0f}","top10_weight_pct":"{:.2f}%"}), use_container_width=True, hide_index=True)
    comp = compare3(q0,q1,q2)
    trends = ["連續加碼","連續減碼","本期新進","新進後續加碼","本期清倉","本期轉加碼","本期轉減碼","股數不變","其他"]
    chosen = st.multiselect("篩選趨勢", trends, default=["連續加碼","本期新進","新進後續加碼","本期清倉"])
    view = comp[comp.trend.isin(chosen)] if chosen else comp
    rename = {"shares_q0":f"shares_{labels[0]}","shares_q1":f"shares_{labels[1]}","shares_q2":f"shares_{labels[2]}","value_q0":f"value_{labels[0]}","value_q1":f"value_{labels[1]}","value_q2":f"value_{labels[2]}","weight_q0":f"weight_{labels[0]}","weight_q1":f"weight_{labels[1]}","weight_q2":f"weight_{labels[2]}"}
    st.markdown("### 三期持股變化")
    st.caption("重點看 shares 變化；value 是期末市值，會受股價漲跌影響。")
    st.dataframe(view.rename(columns=rename).head(max_rows), use_container_width=True, hide_index=True)
    st.download_button("下載三期比較 CSV", comp.rename(columns=rename).to_csv(index=False).encode("utf-8-sig"), file_name=f"13f_3q_{name}_{labels[0]}.csv", mime="text/csv")
    st.markdown("### 本期前十大持股")
    st.dataframe(q0.head(10), use_container_width=True, hide_index=True)
    with st.expander("資料來源 URL"):
        for label, url in zip(labels, urls): st.write(f"{label}: {url}")

if __name__ == "__main__":
    main()
