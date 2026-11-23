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
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

Mac、Linux 或 WSL（Ubuntu）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

執行紀錄會記錄 `pytest` 版本，因此必須安裝 `requirements-dev.txt` 中的固定版本，即使只是執行分析也需要安裝。

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

## Windows 使用者

專案透過 `.gitattributes` 固定文字檔使用 LF，避免 `core.autocrlf` 將檔案轉成 CRLF，
造成 `docs/frozen_rules.md` 或資料檔的 SHA256 與凍結紀錄不符。更新屬性不會自動重寫已存在的工作目錄檔案。

拉下此修正後，先確認以下兩檔沒有需要保留的本機修改，再於專案根目錄用 PowerShell 執行：

```powershell
git pull --ff-only
Remove-Item docs/frozen_rules.md, data/default_of_credit_card_clients.csv
git checkout -- docs/frozen_rules.md data/default_of_credit_card_clients.csv
git check-attr eol -- docs/frozen_rules.md data/default_of_credit_card_clients.csv outputs/frozen/rules-g2a-20261116.json
```

最後一個指令的三行結果都應為 `eol: lf`。重新取回後，再執行原本的規則載入／分析指令。
不要修改凍結紀錄的雜湊來配合 CRLF 檔案；雜湊檢查仍會嚴格比對檔案原始位元組。

## 執行

平台只在本機執行，不要部署到公開網址。在本機 `.env` 設定 `PLATFORM_RUN_ID`，
並與 `PLATFORM_FAIRNESS_RUN_ID` 配套切換。正式設定如下，只在本機設定，不提交或截圖 `.env`。

```dotenv
PLATFORM_RUN_ID=20261122-144512-34512c5d87a3
PLATFORM_FAIRNESS_RUN_ID=20261122-144515-a7bbbea040a4
```

平台會驗證整組凍結模型、政策與 6,000 人交接檔，公平性比較必須來自同一凍結版本及 summary。
正式資料載入失敗會顯示錯誤，不自動退回開發資料。門檻沿用 L1 OOF 的正式政策，不在平台重選。
需要開發與功能測試時，明確指定 `00_build_dev_base.py` 印出的 M1-NS-dev19200 L1-H run，
並搭配 L1 的 F3 比較 run。不得混用模型、門檻或結果。
切換資料不會清除批次歷史或解除同期防重送，舊 run 的通知紀錄不代表新資料已通知。


```bash
python -m streamlit run app/main.py       # 開啟本機平台，瀏覽器打開 http://localhost:8501
cd experiments && python 01_tuning.py     # 執行某個實驗（檔名換成自己的）
```

注意：程式載入 `.env` 時，已存在的環境變數優先。終端機若殘留 `PLATFORM_*`、`LINE_*` 設定，會蓋過 `.env` 的同名設定。啟動平台或批次前，若曾在終端機手動設定，請先開新終端機，或在 WSL／Bash 用 `unset` 清除曾設定的變數。

以下指令只顯示相關變數名稱，不顯示值，沒有符合項目時不輸出：

```bash
env | cut -d= -f1 | grep -E '^(PLATFORM_|LINE_)'
```

例如清除平台設定可用 `unset PLATFORM_RUN_ID PLATFORM_FAIRNESS_RUN_ID`，LINE 設定也依檢查列出的名稱清除。若新終端機仍出現這些名稱，請確認啟動設定是否會重新匯入，不要列印或截圖變數值。

要關閉平台時，在終端機按 Ctrl + C。

## 分工

| 組員 | 工作 | 檔案 | 分支 |
|---|---|---|---|
| 余旻恩 | 調參、重複交叉驗證（RQ1 至 RQ3）、輸出「移除敏感屬性」的第二份交接檔、機率校準檢查 | experiments/01、03、08、src/export.py | exp/tuning、exp/repeated-cv、exp/export-no-sensitive、exp/calibration |
| 陳品婕 | SHAP 分析、繳款狀態類別編碼、漏判違約者分析 | experiments/04～05、09 | exp/shap、exp/pay-categorical、exp/missed-analysis |
| 張佩芩 | 公平性加入教育與婚姻分組、族群門檻（改善公平性）、金管會指引檢核表 | experiments/06、07、docs/ | exp/fairness、exp/group-threshold |
| 蔡翌佳 | 平台「公平性監控」與「單一客戶查詢」頁面、展示影片 | app/views/fairness.py、app/views/customer.py | feat/fairness-page、feat/customer-page |
| 蘇韻蓉 | 門檻與成本分析（含成本假設依據與名單上限）、每月風險名單頁、TAIDE 與改善前後比較、關懷訊息、LINE Bot、整合與審核 | experiments/02、10、app/main.py、app/views/risk_list.py、llm/、notify/ | exp/threshold、exp/taide-eval、feat/platform、feat/taide、feat/line |

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
2. 在本機依 `.env.example` 設定測試客戶與測試行員帳號，分別使用 `LINE_TEST_CUSTOMER_ID` 與 `LINE_TEST_STAFF_ID`，不要提交或分享設定內容。舊名稱 `LINE_TEST_USER_ID` 只相容客戶帳號，使用時會提醒改名。
3. `LINE_SEND_ENABLED` 預設維持 false。`notify.line_push` 與 `notify.staff_digest` 的 CLI 不提供發送功能，批次入口見下節。
4. 客戶訊息仍限固定模板，每次都須確認相同內容，修改、換客戶或發送後都要重新核准。行員摘要只含人數、最多十位客戶的 ID 與排名及平台連結，不需逐則核准，但同樣受總開關控制。
5. 風險名單頁可預覽行員摘要，摘要範圍與瀏覽滑桿無關，名單總人數仍由交接政策決定。平台連結只允許本機、私有測試網段或保留的 `.test` 網域根頁，不建立公開網址。同一帳號僅為「同一測試帳號模擬」，不同帳號才可驗證分別接收的流程。

## 每月批次與 Windows 工作排程器

批次沿用平台 `PLATFORM_RUN_ID` 的完整驗證、政策與交接名單，不訓練模型、不重選門檻。
PR #57 起批次沿用平台的正式 L2 run（`PLATFORM_RUN_ID`），但 L2 是歷史測試資料，批次是每月流程的模擬，不是真實每月新增資料。
在專案根目錄，使用專案虛擬環境執行：

```bash
.venv/bin/python -m app.batch_run --period 2026-11
```

批次只通知測試行員，不發送客戶關懷訊息。這是 C12 明確新增的排程入口，
`notify.line_push` 與 `notify.staff_digest` 原有 CLI 仍不發送。
預設 `LINE_SEND_ENABLED=false`，模擬紀錄不會標成已通知。資料驗證、欄位、非空名單或政策上限檢查
失敗時，紀錄為「異常，待人工確認」，不推播。
每次嘗試的彙總紀錄保存在 `outputs/batches/<YYYY-MM>/<batch_id>.json`，不存逐客戶資料或憑證，且不上傳 Git。
同期已有成功推播就略過，即使更換 PLATFORM_RUN_ID 也一樣。若網路結果不明或程序中斷，不自動重送。
人工確認後才可使用 `--force`，新紀錄保留 force、先前嘗試編號與是否強制重試，舊紀錄不覆寫。
若有 `.lock` 殘留，須先人工確認沒有執行中的程序與未明發送，不要讓排程自動清除。

Windows 工作排程器設定：

1. 先用 `wsl -l -v` 確認發行版名稱，並確認專案在 WSL 內的絕對路徑與 `.venv/bin/python` 可執行。
2. 批次命令列入口與平台一樣，先載入專案根目錄的 `.env`，再讀取 `PLATFORM_RUN_ID` 與 LINE 設定。
   依 `.env.example` 在專案 `.env` 設定，不需在 WSL 登入環境另外設定。已存在的程序環境變數優先，不會被覆寫。
   `.env` 不得提交、截圖或分享，不把設定值放進排程命令。總開關保持 false 做功能驗證，未設定會安全地留下異常紀錄。
   一般匯入 `app.batch_run` 不會載入 `.env`。
3. 開啟「工作排程器」→「建立工作」，設定每月固定日期與時間觸發。選擇已設定 WSL 的同一個 Windows 使用者執行。
4. 動作選「啟動程式」，程式填 `wsl.exe`，引數如下。將 Ubuntu、專案路徑換成本機實際值，保留引號。
   日期由 WSL 的台北時區計算，每次取得當月份，不要將期別寫死。此引數填在排程器內，不在 PowerShell 直接展開。

   ```text
   -d Ubuntu --cd /home/user/credit-warning --exec bash -lc "exec .venv/bin/python -m app.batch_run --period $(TZ=Asia/Taipei date +%Y-%m)"
   ```

5. 「設定」選擇工作已執行時不要啟動新的執行個體，不設定失敗後自動重試，不在排程加入 `--force`。
   可勾選錯過排程後儘快執行，但期別會以實際執行月份為準，補跑舊月份需人工明確指定。
6. 電腦必須開機，睡眠或關機會使工作延後或無法執行。若需從睡眠喚醒，可依電腦政策啟用喚醒選項，關機無法喚醒執行。
   排程器的結束碼 0 表示成功、模擬或已通知而略過，1 表示異常，2 表示命令參數錯誤。
7. 執行後在平台「最近一次批次」檢查期別、時間、人數、檢查結果與是否通知。正式啟用前由組長確認設定，
   本工作卡的驗證只使用假設定與 Mock，沒有實際對外發送。
