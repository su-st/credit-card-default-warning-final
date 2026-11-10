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

需要：Python 3.10 以上（建議 3.12）、Git。

### 1. 下載專案

```bash
git clone https://github.com/su-st/credit-card-default-warning-final.git
cd credit-card-default-warning-final
```

### 2. 建立虛擬環境並安裝套件

虛擬環境 `.venv` 只給這個專案用，不會跟其他課的套件衝突。

Windows（cmd）：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Mac、Linux 或 WSL（Ubuntu）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

啟用成功時，終端機開頭會出現 `(.venv)`。**之後每次打開專案，都要先執行第二行啟用虛擬環境。**

### Windows 11 安裝被擋住怎麼辦

如果出現「已被貴組織的 Device Guard 原則封鎖」或「應用程式控制原則已封鎖此檔案」，
代表 Windows 的「智慧型應用程式控制」擋住了 Python 套件。

**不要關閉智慧型應用程式控制**，關掉後可能要重設電腦才能再打開。請改用 WSL（Windows 內建的 Linux 環境）：

1. 在 cmd 輸入 `wsl -l -v`，有看到 `Ubuntu` 就代表已經裝過，跳到第 3 步
2. 沒有的話，以系統管理員身分開啟 PowerShell，輸入 `wsl --install`，完成後重新開機並設定 Linux 帳號密碼
3. 在 cmd 輸入 `wsl` 進入 Ubuntu，再輸入：
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-pip git
   cd ~
   ```
4. 接著從上面「1. 下載專案」開始，照「Mac、Linux 或 WSL」的指令做

如果 Ubuntu 裡出現 `Temporary failure resolving` 或 `Could not resolve host`，
通常是防毒軟體擋住了 WSL 的網路。安裝套件、`git push`、`git pull` 時暫時關閉防毒，完成後再打開。
還是不行就問組長。

## 執行

```bash
python -m src.export                      # 1. 產生交接檔 outputs/predictions.csv（約 10 秒）
python -m streamlit run app/main.py       # 2. 開啟平台，瀏覽器打開 http://localhost:8501
cd experiments && python 01_tuning.py     # 執行某個實驗（檔名換成自己的）
```

要關閉平台時，在終端機按 Ctrl + C。

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
git clone https://github.com/su-st/credit-card-default-warning-final.git
cd credit-card-default-warning-final
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
