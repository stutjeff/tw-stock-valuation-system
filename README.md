# 台股錯殺・轉機・護城河估值系統

這是一個可部署在 Streamlit Cloud 的手機版台股研究工具。

## 功能

輸入台股股票代號後，自動產出：

1. 公司名稱、產業分類、市場別
2. 最新股價、PE、PB、ROE、負債比、月營收 YoY、法人買賣超
3. 20 / 60 / 240 日布林通道
4. PE / PB 歷史分位
5. 月營收與年增率
6. 三大法人買賣超
7. 融資融券
8. 財務體質摘要
9. 三類股票判斷：
   - 基本面沒問題被錯殺
   - 轉機股
   - 穩定有護城河且股價合理
10. 轉機/錯殺/護城河追蹤清單
11. Google Trends：台灣/美國熱門搜尋、關鍵字搜尋熱度比較、相關搜尋/飆升搜尋
12. 可選 AI 產業賽道、護城河、競爭對手與風險分析

## 使用資料源

主要使用 FinMind 免費 API：

- TaiwanStockInfo
- TaiwanStockPrice
- TaiwanStockPER
- TaiwanStockMonthRevenue
- TaiwanStockInstitutionalInvestorsBuySell
- TaiwanStockMarginPurchaseShortSale
- TaiwanStockFinancialStatements
- TaiwanStockBalanceSheet
- TaiwanStockCashFlowsStatement
- TaiwanStockDividend

FinMind 不填 token 也可使用，但有請求限制。建議部署後在 Streamlit Secrets 加入 FINMIND_TOKEN。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 部署

1. 建立 GitHub repository
2. 上傳以下檔案：
   - app.py
   - requirements.txt
   - README.md
3. 到 Streamlit Cloud 新增 App
4. Repository 選你的專案
5. Main file path 填：

```text
app.py
```

6. Deploy

## Streamlit Secrets

在 Streamlit Cloud：

```text
App → Settings → Secrets
```

可填入：

```toml
FINMIND_TOKEN = "你的 FinMind Token"
OPENAI_API_KEY = "你的 OpenAI API Key"
OPENAI_MODEL = "gpt-4o-mini"
```

FINMIND_TOKEN 可選。OPENAI_API_KEY 可選。

沒有 OPENAI_API_KEY 時，系統仍能運作，只是 AI 質化分析會改用規則版。

## Google Trends 功能

新增 `Google Trends` 分頁：

- 台灣 / 美國熱門搜尋榜
- 最多 5 個關鍵字的台灣 / 美國搜尋熱度比較
- 單一關鍵字的 Top / Rising 相關搜尋

注意：Google Trends 顯示的是相對搜尋熱度，不是實際搜尋量。100 代表所選地區與期間內最高熱度。

目前使用 `pytrends` 這個非官方 Google Trends 套件；Google 若調整後端，資料可能暫時讀取失敗。

## 注意

本工具是研究輔助，不是買賣建議。

台股中小型股的市佔率、客戶名單、產能利用率、許可進度、法說會內容，常常無法只靠 API 完整判斷。這些資料仍需要搭配年報、公開資訊觀測站、法說會、新聞與公司公告人工確認。
