# 可解釋且公平的信用卡違約預警平台

人工智慧期末專題。使用 UCI Default of Credit Card Clients 資料（CC BY 4.0），
完成三個研究問題的實驗，並做出以歷史資料與模擬帳號展示的預警平台原型。

## 資料夾結構

```
src/            共用程式（資料、特徵、模型、評估、交接檔匯出），改動前先告知組長
experiments/    各組員的實驗，一人一檔，互不干擾
app/            Streamlit 平台（main.py 與三個頁面 views/）
llm/            TAIDE 說明與固定句型備案
notify/         LINE 推播
data/           UCI 原始資料
docs/           交接檔欄位說明等文件
outputs/        執行後產生的結果（不上傳 GitHub）
```

## 安裝

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows 用 copy .env.example .env
```

## 執行

```bash
python -m src.export              # 1. 產生交接檔 outputs/predictions.csv（約 10 秒）
streamlit run app/main.py         # 2. 開啟平台
cd experiments && python 01_tuning.py   # 執行某個實驗
```

## 分工

| 組員 | 工作 | 檔案 | 分支 |
|---|---|---|---|
| 余旻恩 | 調參、重複交叉驗證（RQ1 至 RQ3）、輸出「移除敏感屬性」的第二份交接檔 | experiments/01、03、src/export.py | exp/tuning、exp/repeated-cv、exp/export-no-sensitive |
| 陳品婕 | SHAP 分析、繳款狀態類別編碼 | experiments/04～05 | exp/shap、exp/pay-categorical |
| 張佩芩 | 公平性加入教育與婚姻分組、金管會指引檢核表 | experiments/06、docs/ | exp/fairness |
| 蔡翌佳 | 平台「公平性監控」與「單一客戶查詢」頁面、展示影片 | app/views/fairness.py、app/views/customer.py | feat/fairness-page、feat/customer-page |
| 蘇韻蓉 | 門檻與成本分析、每月風險名單頁、TAIDE、關懷訊息、LINE Bot、整合與審核 | experiments/02、app/main.py、app/views/risk_list.py、llm/、notify/ | exp/threshold、feat/platform、feat/taide、feat/line |

## Git 分支流程

`main` 只放可以正常執行的版本，每個人在自己的分支上做，完成後發 Pull Request 給組長合併。

```bash
git clone <儲存庫網址>
cd credit-warning
git checkout -b exp/tuning            # 開自己的分支（名稱見上表）

# 修改、執行、確認沒問題後
git add experiments/01_tuning.py
git commit -m "完成 XGBoost 調參"
git push -u origin exp/tuning         # 到 GitHub 按 Compare & pull request
```

開始工作前先同步 main 的最新版本：

```bash
git checkout main && git pull
git checkout exp/tuning && git merge main
```

規則：
1. 只改自己負責的檔案，要改 `src/` 或別人的檔案先在群組說
2. 不要上傳 `.env`（裡面有 LINE token）和 `outputs/`
3. 實驗結果存到 `outputs/results/`，報告要用的表格和圖再另外放到共用雲端
4. 所有實驗固定 `random_state=42`，使用 `src/data.py` 的切分，確保大家的數字一致

## TAIDE（本機）

1. 到 Hugging Face 的 TAIDE 頁面確認授權條款並申請下載
2. 安裝 Ollama，匯入 TAIDE 模型（建議 8B 的 4-bit 量化版本）
3. 在 `.env` 填入 `TAIDE_MODEL`（`ollama list` 顯示的名稱）
4. 沒有設定或呼叫失敗時，平台自動改用 `llm/templates.py` 的固定句型

## LINE Bot

1. 在 LINE Developers 建立 Messaging API channel
2. 在 `.env` 填入 Channel access token 和測試帳號的 User ID
3. 執行 `python -m notify.line_push` 測試，手機應該收到一則訊息
4. 平台上必須勾選「我已確認訊息內容正確」才能發送，只會發到測試帳號
