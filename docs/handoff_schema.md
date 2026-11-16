# 交接檔與 C1 登錄契約

每個 run 使用 `outputs/runs/<run_id>/predictions.csv`，不再讀寫全域
`outputs/predictions.csv`、`meta.json` 或 `threshold.json`。模型與政策分開識別。
欄位沿用舊平台名稱，新增識別欄位。平台載入與快取由 C6 遷移；
完成前不可把新檔複製成舊檔名混用。

| 欄位 | 說明 |
|---|---|
| customer_id | UCI 原始 ID，唯一、正整數 |
| risk_score | 模型風險分數，0～1；不是違約機率。CSV 保留完整雙精度，畫面才四捨五入 |
| risk_rank | 完整分數降冪、同分依 customer_id 升冪，從 1 起算 |
| in_list | 同次 run 的政策套用於 risk_score 的最終名單；有上限時依排序截取 |
| sex / age / age_group / education / marriage | 公平性監控用族群資訊 |
| limit_bal / bill_amt_latest | 信用額度／最近一期帳單金額 |
| actual_default | 實際標籤，只供歷史回測；新資料預警模式不顯示 |
| data_split | C1 為 dev_holdout（L1-H，開發資料，非正式結果）；L2 由凍結後的 final_eval 產生 |
| factor_1～3 | TreeSHAP 推高分數的前三個模型判斷依據，不代表違約原因；One-Hot 帶正負號合併後再選取 |
| model_id / policy_id / run_id | 模型、名單政策與執行識別，所有列相同 |

## 產物位置與驗證

- `outputs/models/<model_id>-<model_hash前12碼>/`：`model.joblib`、`model.json`、
  `oof.csv`（customer_id、oof_score、y）、`folds.csv`（customer_id、fold）。
  同名但訓練結果不同的模型存到不同目錄，相同模型檔不重複保存。
- `model.json`：模型檔 SHA256、完整參數與特徵清單／雜湊、資料 SHA256、
  訓練 ID 雜湊、人數、類別權重算法、資料層級、OOF 與折分配雜湊。
- `threshold.json`：policy_id／hash、model_id／hash、run_id、params_hash、
  features_hash、training_ids_hash、seed、n_splits、folds_sha256、oof_sha256、
  rules_freeze_id、source_level、成本情境及名單政策內容。
- `run.json`：狀態、資料層級、Git 狀態、套件版本、資料／模型／政策識別、
  predictions_sha256、重新載入驗證的人數與結果。`results/<run_id>/run.json`
  是同一紀錄副本，彙總 CSV 與圖也在這裡。

`Registry.load_run(run_id)` 檢查三份檔案識別、模型／OOF／折分配／交接檔雜湊
及名單判定。一項不符就停止；缺檔不會默默退回 0.5。
`load_threshold(allow_fallback=True)` 只在沒有指定 run 時明確回傳備用值，標示
「未使用正式門檻」；損壞的既有 run 仍報錯。平台不會自動啟用。
平台快取應以 run_id 與交接檔雜湊作鍵（C6）。

## C1 的資料與執行界線

`Registry.register_model` 接受已訓練模型、實際 OOF 與對齊折分配，不訓練、不挑參數。
L1-H 模型另外記錄開發保留集 ID 雜湊，與訓練 ID 不得重疊。
`start_run` 可建立 L0／L1／L1-H；L2 必須讀取已存在且有組長核准紀錄的凍結項目，
沿用該項目引用的 L1 模型與政策，禁止重訓或混用 L1-H。
第 4.4 節公開 protocol、正式資料存取與 L3／L4 執行由 C2 完成；
C1 不建立凍結紀錄、不選正式設定。

凍結紀錄採 JSON：rules-<id>.json 含 rules_freeze_id 與 markdown_sha256；
model-<id>.json 含 model_freeze_id、rules_freeze_id、approved_at、entries。
entries 每筆含 entry_id、role、comparison_id、source_run_id（已完成 L1 run）、
model_id、model_hash、params_hash、columns、policy_id、policy_hash。
讀取時核對來源證據與 24,000 筆訓練範圍，run 另記錄凍結檔雜湊。
這是規格的程序控制，不是密碼學簽章；凍結建立與組長核准不由 C1 自動執行。

先建立 L1-H run、綁定登錄模型，再透過
`save_threshold(info, run=run, model_record=record)` 寫入該模型自己的 OOF 門檻，然後：

```bash
python -m src.export --run-id <新的running狀態L1-H-run>
```

export 使用登錄的特徵與模型，不重新訓練、不讀平台 config 決定敏感屬性。
M1-dev 與 M1-NS-dev 各有自己的模型、OOF、政策與 run，都是開發版本，
不得使用正式模型的門檻。C1 暫用 src.data 既有切分取得 L1-H，
C2 再將此資料來源換為 protocol.dev_holdout。
export 重新載入模型，核對全部 4,800 人的分數（絕對誤差 ≤1e-9），通過才完成。
拒絕 test／L2 等資料層級，即使呼叫者傳入資料也不預測。

02 從已登錄的 L1 模型讀 OOF，原平均損失公式、選門檻算法與敏感度計算不變，
只遷移讀寫位置並綁定證據，每個敏感度情境各有 run 與政策。
設定從 rules_freeze 的 legacy_threshold_analysis 讀取，這是舊算法對照，
C3 的正式 D2 政策另行實作。02 要等 C2 的 dev_data 與規則凍結紀錄就緒才能執行，
不退回模組匯入時就取得測試集的舊 _common。
legacy_threshold_analysis 包含 loss_per_default、cost_per_contact、rescue_rate、
rule、top_n_rate、max_list_rate、sensitivity_contact_costs、sensitivity_rescue_rates；
本卡不建立或自行填入凍結數值。設定只換成 JSON 輸入，選擇公式不變。

分群模型用 GroupedModel 封裝已訓練子模型、PAY 正值分組規則與特徵清單。
只封裝 F6-D 的既有路由，不修改組員的分析與訓練邏輯。
執行失敗標示 failed，不可當正式結果；重跑用新 run_id。
run 目錄已存在即拒絕建立，完成後不再寫入。模型、OOF、逐客戶交接檔與 outputs 不進 Git。
