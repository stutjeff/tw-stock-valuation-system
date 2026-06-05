# 全球錯殺・轉機・護城河估值系統 V2

支援：

- 台股：FinMind 資料源
- 美股：yfinance 資料源
- 日股：yfinance 資料源，輸入 7203 會自動轉成 7203.T

## 新增功能

1. 系統自動勾選追蹤清單
2. 台股 / 美股 / 日股三市場切換
3. 美股與日股支援股價布林通道、PE、PB、ROE、毛利率、營益率、現金流、營收成長率
4. 保留 AI 質化分析擴充入口

## Streamlit 部署設定

- Repository：你的 GitHub repo
- Branch：main
- Main file path：app.py
- Python version：3.11

## 建議 Secrets

```toml
FINMIND_TOKEN = "你的 FinMind Token，可不填"
OPENAI_API_KEY = "你的 OpenAI API Key，可不填"
OPENAI_MODEL = "gpt-4o-mini"
```

沒有 OPENAI_API_KEY 時，系統會使用規則版分析。

## 上傳檔案

請上傳：

- app.py
- requirements.txt
- README.md
- runtime.txt
- .python-version（iPhone 可能隱藏，看不到可略過）

