# AGENTS.md（給 Codex）

## 專案
信用卡違約預警：可解釋且公平的模型與預警平台原型。規格以 docs/final_spec.md 為準，衝突時以規格為準並回報。

## 一定要做
- 讀資料、切分、特徵、評估一律用 src/ 與 experiments/_common.py，不要自己重寫。
- 每個結果都記錄 model_id、policy_id 與 run_id。模型與政策分開識別，政策只能來自同一次執行的 threshold.json 或凍結紀錄。
- 介面名稱、參數與回傳欄位照 docs/final_spec.md 第 4.4 節實作，不可自行更改。設定一律從凍結紀錄讀取。
- 設計決策（參數、門檻、特徵、分群、成本規則）只能用 L1（seed 42 訓練集 OOF）。平台開發用 L1-H。
- 狀態為完成的 outputs/runs/<run_id>/ 與 outputs/frozen/ 的凍結紀錄不可覆寫，重跑就產生新的 run_id。
- L1-H 只用於介面、串接與功能測試，不可依 L1-H 的指標調整任何設定，也不可混用 L1-H 與正式版本的模型、門檻或 run_id。
- 每個 PR 只做一張工作卡，PR 說明附驗收結果與 pytest 結果。
- 對不上期中數字時停下來回報，不改程式湊數字。

## 不可以
- 不讀、不印、不上傳 .env 的內容。不把 outputs/、模型檔、data/ 加進 git。
- G2b 模型凍結前，不對 seed 42 測試集產生或讀取任何預測、名單、SHAP 或指標（L0 的期中重現除外，且不得依 L0 結果修改任何設計）。
- 不修改 docs/frozen_rules.md 已凍結的內容，除非組長要求並記錄原因。
- LINE 發送只能用 send_customer_care 與 send_staff_digest，不接受指定收件者。
- 不修改期中結果與第 7 版報告的數字。
- 不自行決定正式模型、門檻、成本假設、名單上限或 G2b 條件數值。
- 不把 LINE_SEND_ENABLED 預設改成 true，不放寬 PR #8 的客戶訊息限制。
- 不把 PAY 代碼 0、−2 的意思寫成確定事實。
- 不把風險分數寫成違約機率，不把 SHAP 因素寫成違約原因。
- 不改組員負責的分析邏輯。

## 審查組員 PR
- 可以直接修（先確認組員同意，另開 commit 並留言）：格式、命名、錯字、路徑寫死、缺少換行、未使用的 import。
- 只能留言：資料層級與測試集使用、模型與門檻版本、指標與分母計算、統計解讀、結論用詞。
- 檢查清單：
  1. 有沒有用 _common 與 protocol？
  2. 有沒有在 G2b 模型凍結前碰到測試集？
  3. model_id、run_id 與門檻是否一致？
  4. 結果是否輸出到 results/<run_id>/？
  5. 族群是否同時列出總人數、違約人數、未違約人數？分母為 0 是否標示？
  6. 提前預警的特徵是否只用到可取得的月份？
  7. 用詞是否把風險分數寫成機率、把關聯寫成原因？
  8. AI 工具使用是否在 PR 說明揭露？

## 環境
套件版本見 requirements.txt 與 requirements-dev.txt，測試指令 python -m pytest -q。TAIDE 經由本機 Ollama（OLLAMA_URL、TAIDE_MODEL），沒有 TAIDE 時測試也必須能跑。
